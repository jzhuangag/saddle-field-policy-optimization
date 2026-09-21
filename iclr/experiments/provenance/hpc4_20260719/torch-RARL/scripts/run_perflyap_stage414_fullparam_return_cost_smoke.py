from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, named_parameters, restore_state
from scripts.run_perflyap_stage4_preflight import build_model, pair_probes, role_algo
from scripts.run_perflyap_stage412_actual_merit_oracle import (
    actual_merit_evaluator as _unused_actual_merit_evaluator,
    baseline_one_step_rows,
    direction_maps,
    evaluate_oracle_candidates,
    evaluate_state,
    instantiate_helper,
    load_baseline_settings,
    metric_row as _unused_metric_row,
    rollout_return_with_model,
)
from scripts.run_perflyap_stage48_perf_audit import build_eval_context, build_extended_eval_closure


@dataclass(frozen=True)
class FullParamConfig:
    config_id: int
    scope: str
    lambda_N: float
    cost_mode: str
    beta_max: float
    gamma_max: float
    update_cap: float

    @property
    def label(self) -> str:
        return f"{self.scope}_n{self.lambda_N:g}_{self.cost_mode}_b{self.beta_max:g}_g{self.gamma_max:g}_cap{self.update_cap:g}"


MERIT_NAMES = [
    "mixed_clean_actor_surrogate_cost",
    "mixed_clean_unclipped_actor_surrogate_cost",
    "mixed_rarl_actor_surrogate_cost",
    "mixed_rarl_unclipped_actor_surrogate_cost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.14 full-param return+cost smoke")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--baseline-summary", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--num-probes", type=int, default=2)
    parser.add_argument("--beta-points", type=int, default=11)
    parser.add_argument("--gamma-points", type=int, default=11)
    parser.add_argument("--short-horizon", type=int, default=16)
    parser.add_argument("--return-episodes", type=int, default=1)
    parser.add_argument("--adv-strength", type=float, default=1.0)
    args = parser.parse_args()
    # Keep these helper args aligned with Stage 4/4.12 utilities.
    args.num_probes_per_role = max(int(args.num_probes), 4)
    args.max_configs = 1
    args.lr = 1.0
    args.max_grad_norm = 1.0
    args.vf_coef = 1.0
    return args


def build_configs() -> List[FullParamConfig]:
    configs: List[FullParamConfig] = []
    config_id = 0
    for scope in ["full_policy_actor_weighted", "actor_mean_heavy", "critic_downweighted"]:
        for cost_mode in ["actor_surrogate_cost", "unclipped_actor_surrogate_cost"]:
            configs.append(
                FullParamConfig(
                    config_id=config_id,
                    scope=scope,
                    lambda_N=0.0,
                    cost_mode=cost_mode,
                    beta_max=3e-2,
                    gamma_max=3e-5,
                    update_cap=0.005,
                )
            )
            config_id += 1
    return configs


def mixed_merit_evaluator(
    merit_name: str,
    *,
    helper,
    eval_closure,
    selected_names: Sequence[str],
    model,
    env_id: str,
    episodes: int,
    horizon: int,
    base_seed: int,
    adv_strength: float,
):
    def eval_theta(theta_state):
        named_params = named_parameters(model.protagonist.policy)
        restore_state(named_params, theta_state)
        state_info = evaluate_state(helper, eval_closure, theta_state, selected_names)
        if "unclipped" in merit_name:
            actor_cost = float(state_info["policy_unclipped_component"])
        else:
            actor_cost = float(state_info["policy_component"])
        clean_ret, _ = rollout_return_with_model(
            model,
            env_id=env_id,
            adv_strength=0.0,
            episodes=episodes,
            horizon=horizon,
            base_seed=base_seed,
            control_adv=False,
        )
        clean_cost = -float(clean_ret)
        if merit_name.startswith("mixed_rarl_"):
            adv_ret, _ = rollout_return_with_model(
                model,
                env_id=env_id,
                adv_strength=adv_strength,
                episodes=episodes,
                horizon=horizon,
                base_seed=base_seed,
                control_adv=True,
            )
            rarl_cost = -0.5 * float(clean_ret) - 0.5 * float(adv_ret)
            total = actor_cost + rarl_cost
            return {"merit": total, "actor_cost": actor_cost, "return_cost": rarl_cost, "approx_kl": float(state_info["approx_kl"]), "clip_fraction": float(state_info["clip_fraction"])}
        total = actor_cost + clean_cost
        return {"merit": total, "actor_cost": actor_cost, "return_cost": clean_cost, "approx_kl": float(state_info["approx_kl"]), "clip_fraction": float(state_info["clip_fraction"])}

    return eval_theta


def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["config_id", "config_label", "scope", "lambda_N", "cost_mode", "merit_name", "method", "direction"]
    for keys, group in detail_df.groupby(group_cols):
        config_id, config_label, scope, lambda_N, cost_mode, merit_name, method, direction = keys
        rows.append(
            {
                "config_id": int(config_id),
                "config_label": config_label,
                "scope": scope,
                "lambda_N": float(lambda_N),
                "cost_mode": cost_mode,
                "merit_name": merit_name,
                "method": method,
                "direction": direction,
                "rows": int(len(group)),
                "actual_merit_change_mean": float(group["actual_merit_change"].mean()),
                "actual_merit_change_min": float(group["actual_merit_change"].min()),
                "approx_kl_max": float(pd.to_numeric(group["approx_kl"], errors="coerce").max()),
                "clip_fraction_max": float(pd.to_numeric(group["clip_fraction"], errors="coerce").max()),
                "gamma_active_mean": float(pd.to_numeric(group["gamma_active"], errors="coerce").mean()),
                "fallback_to_noG_sum": int(pd.to_numeric(group["fallback_to_noG"], errors="coerce").fillna(0).sum()),
                "actor_update_norm_mean": float(pd.to_numeric(group["actor_update_norm"], errors="coerce").mean()),
                "logstd_update_norm_mean": float(pd.to_numeric(group["logstd_update_norm"], errors="coerce").mean()),
                "critic_update_norm_mean": float(pd.to_numeric(group["critic_update_norm"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    focus_methods = ["adam", "sgd", "egm", "ppm", "noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"]
    focus = summary_df[summary_df["method"].isin(focus_methods)].copy()
    focus = (
        focus.groupby(["merit_name", "method"], as_index=False)
        .agg({"actual_merit_change_mean": "mean"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for merit_name, group in focus.groupby("merit_name"):
        ordered = group.set_index("method").reindex(focus_methods)
        ax.plot(np.arange(len(focus_methods)), ordered["actual_merit_change_mean"], marker="o", label=merit_name)
    ax.set_xticks(np.arange(len(focus_methods)))
    ax.set_xticklabels(focus_methods, rotation=25, ha="right")
    ax.set_ylabel("Actual merit change")
    ax.set_title("Stage 4.14 full-param one-step actual merit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage414_methods_one_step_actual_merit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for merit_name, group in focus.groupby("merit_name"):
        sub = group[group["method"].isin(["noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"])]
        ordered = sub.set_index("method").reindex(["noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"])
        ax.plot(np.arange(3), ordered["actual_merit_change_mean"], marker="o", label=merit_name)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"], rotation=20, ha="right")
    ax.set_ylabel("Actual merit change")
    ax.set_title("Stage 4.14 QP vs noG actual merit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage414_qp_vs_nog_actual_merit.png", dpi=180)
    plt.close(fig)


def write_report(summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    grouped = (
        summary_df.groupby(["merit_name", "method", "direction"], as_index=False)
        .agg(
            {
                "actual_merit_change_mean": "mean",
                "gamma_active_mean": "mean",
                "actor_update_norm_mean": "mean",
                "logstd_update_norm_mean": "mean",
                "critic_update_norm_mean": "mean",
            }
        )
    )
    lines = [
        "# Stage 4.14 Full-Param Return+Cost Smoke Report",
        "",
        "- Scope family: `full_policy_actor_weighted`, `actor_mean_heavy`, `critic_downweighted`.",
        "- Merit is `return cost + surrogate cost` with `lambda_N = 0`.",
        "- This is same-start actual-oracle smoke only, not online training.",
        "",
    ]
    for merit_name in MERIT_NAMES:
        sub = grouped[grouped["merit_name"] == merit_name]
        lines.append(f"## {merit_name}")
        lines.append("")
        cols = ["method", "direction", "actual_merit_change_mean", "gamma_active_mean", "actor_update_norm_mean", "logstd_update_norm_mean", "critic_update_norm_mean"]
        lines.append("```text")
        lines.append(sub[cols].to_string(index=False))
        lines.append("```")
        lines.append("")

    def merit_mean(merit_name: str, method: str) -> float:
        sub = grouped[(grouped["merit_name"] == merit_name) & (grouped["method"] == method)]
        return float(sub["actual_merit_change_mean"].mean()) if not sub.empty else float("nan")

    lines.append("## Key Answers")
    lines.append("")
    for merit_name in MERIT_NAMES:
        q = merit_mean(merit_name, "qp_oracle_best")
        n = merit_mean(merit_name, "noG_oracle_best")
        lines.append(f"- `{merit_name}`: QP oracle {'beats' if q <= n else 'does not beat'} noG oracle (`{q:.6g}` vs `{n:.6g}`).")
    (output_root / "stage414_fullparam_return_cost_smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_settings = load_baseline_settings(pathlib.Path(args.baseline_summary))
    configs = build_configs()
    detail_rows: List[Dict[str, object]] = []

    model = build_model(args)
    protagonist_algo = role_algo(model, "protagonist")
    probes = collect_probe_batches(model, "protagonist", max(args.num_probes, 4))
    probe_pairs = pair_probes(probes)
    if not probe_pairs:
        raise RuntimeError("Need at least one train/validation probe pair for Stage 4.14.")

    for config in configs:
        for train_probe, val_probe in probe_pairs[: max(1, args.num_probes // 2)]:
            named_params = named_parameters(protagonist_algo.policy)
            helper = instantiate_helper(named_params, config, args)
            train_eval = build_extended_eval_closure(build_eval_context(protagonist_algo, train_probe.rollout_data), config.cost_mode)
            theta_old = clone_state(named_params)
            selected_names = helper._selected_names(named_params)
            train_state_before = evaluate_state(helper, train_eval, theta_old, selected_names)
            merit_names = [
                "mixed_clean_actor_surrogate_cost" if config.cost_mode == "actor_surrogate_cost" else "mixed_clean_unclipped_actor_surrogate_cost",
                "mixed_rarl_actor_surrogate_cost" if config.cost_mode == "actor_surrogate_cost" else "mixed_rarl_unclipped_actor_surrogate_cost",
            ]

            train_eval_ctx, base_train, f_raw_map, direction_dict = direction_maps(
                helper,
                protagonist_algo,
                train_probe,
                theta_old,
                selected_names,
                config,
                args,
            )
            filtered_directions = {k: v for k, v in direction_dict.items() if k in {"egm_minus_JF_F", "egm_actual_delta_direction", "performance_grad", "merit_grad"}}

            for merit_name in merit_names:
                merit_eval = mixed_merit_evaluator(
                    merit_name,
                    helper=helper,
                    eval_closure=train_eval,
                    selected_names=selected_names,
                    model=model,
                    env_id=args.env,
                    episodes=args.return_episodes,
                    horizon=args.short_horizon,
                    base_seed=args.seed + int(train_probe.probe_idx) * 1000,
                    adv_strength=args.adv_strength,
                )
                detail_rows.extend(
                    evaluate_oracle_candidates(
                        helper=helper,
                        named_params=named_params,
                        theta_old=theta_old,
                        selected_names=selected_names,
                        merit_name=merit_name,
                        merit_eval=merit_eval,
                        train_state_before=train_state_before,
                        config=config,
                        train_probe_id=int(train_probe.probe_idx),
                        val_probe_id=int(val_probe.probe_idx),
                        f_raw_map=f_raw_map,
                        direction_dict=filtered_directions,
                        beta_points=args.beta_points,
                        gamma_points=args.gamma_points,
                    )
                )
                detail_rows.extend(
                    baseline_one_step_rows(
                        model=model,
                        algo=protagonist_algo,
                        helper=helper,
                        named_params=named_params,
                        theta_old=theta_old,
                        selected_names=selected_names,
                        merit_name=merit_name,
                        merit_eval=merit_eval,
                        train_state_before=train_state_before,
                        config=config,
                        train_probe=train_probe,
                        val_probe=val_probe,
                        args=args,
                        baseline_settings=baseline_settings,
                    )
                )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = summarize(detail_df)
    detail_df.to_csv(output_root / "stage414_fullparam_return_cost_smoke_detail.csv", index=False)
    summary_df.to_csv(output_root / "stage414_fullparam_return_cost_smoke_summary.csv", index=False)
    plot_results(summary_df, output_root)
    write_report(summary_df, output_root)


if __name__ == "__main__":
    main()
