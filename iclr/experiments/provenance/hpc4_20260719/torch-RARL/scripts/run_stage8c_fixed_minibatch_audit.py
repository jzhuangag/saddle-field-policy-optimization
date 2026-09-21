from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_optimizer_probe import collect_probe_batches
from utils.exp_manager import ExperimentManager
from models.optimizers import block_norm, clone_named_state, named_difference, get_optimizer_class


@dataclass(frozen=True)
class Candidate:
    label: str
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 8C fixed-minibatch audit for full_policy vs actor_game")
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes", type=int, default=3)
    parser.add_argument("--qp-g-alpha", type=float, default=0.3)
    parser.add_argument("--qp-eps", type=float, default=1e-8)
    return parser.parse_args()


def load_stage5_configs(stage5_root: pathlib.Path) -> Dict[str, Dict[str, object]]:
    return json.loads((stage5_root / "final_full_policy_optimizer_configs.json").read_text(encoding="utf-8"))


def choose_ppm_config(configs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    ppm_cfg = configs["ppm"]
    finalists = list(ppm_cfg.get("finalists", []))
    for candidate in finalists:
        inner_steps = int(candidate.get("optimizer_kwargs", {}).get("inner_steps", 1))
        if inner_steps > 2:
            return candidate
    return ppm_cfg


def build_manager(args: argparse.Namespace, scope: str) -> ExperimentManager:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    results_root = repo_root.parent / "results" / "_stage8_probe_tmp" / scope
    ns = argparse.Namespace()
    return ExperimentManager(
        args=ns,
        algo="rarl",
        rarl_config="ppo",
        env_id=args.env,
        log_folder=str(results_root / "logging"),
        tensorboard_log=str(results_root / "tb"),
        n_timesteps=1,
        eval_freq=-1,
        n_eval_episodes=1,
        save_freq=-1,
        hyperparameter_path=str(repo_root / "hyperparameter"),
        hyperparams=None,
        env_kwargs=None,
        model_path=str(results_root / "saved_models"),
        pretrained_model="",
        optimize_hyperparameters=False,
        storage=None,
        study_name=None,
        n_opt_trials=1,
        n_jobs=1,
        sampler="tpe",
        pruner="median",
        optimization_log_path=str(results_root / "opt"),
        n_startup_trials=0,
        n_evaluations_opt=1,
        seed=args.seed,
        log_interval=-1,
        save_replay_buffer=False,
        verbose=0,
        vec_env_type="dummy",
        n_envs=1,
        n_eval_envs=1,
        no_optim_plots=True,
        adv_env=False,
        adv_impact="force",
        adv_fraction=2.5,
        adv_delay=-1,
        adv_index_list=["torso"],
        adv_force_dim=2,
        N_mu=-1,
        N_nu=-1,
        device=args.device,
        protagonist_optimizer="adam",
        adversary_optimizer="adam",
        optimizer_scope=scope,
    )


def optimizer_param_list(optimizer) -> List[th.nn.Parameter]:
    params: List[th.nn.Parameter] = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def make_candidates(configs: Dict[str, Dict[str, object]], scope: str, args: argparse.Namespace) -> List[Candidate]:
    ppm_cfg = choose_ppm_config(configs)
    egm_cfg = configs["egm"]
    sgd_cfg = configs["sgd"]
    proposed_common = {
        "optimizer_scope": scope,
        "qp_g_alpha": args.qp_g_alpha,
        "qp_eps": args.qp_eps,
    }
    return [
        Candidate(
            label=f"{scope}__sgd",
            method="sgd",
            optimizer="sgd",
            lr=float(sgd_cfg["lr"]),
            max_grad_norm=float(sgd_cfg["max_grad_norm"]),
            vf_coef=float(sgd_cfg["vf_coef"]),
            optimizer_kwargs=dict(sgd_cfg.get("optimizer_kwargs", {})),
        ),
        Candidate(
            label=f"{scope}__egm",
            method="egm",
            optimizer="egm",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs=dict(egm_cfg.get("optimizer_kwargs", {})),
        ),
        Candidate(
            label=f"{scope}__ppm",
            method="ppm",
            optimizer="ppm",
            lr=float(ppm_cfg["lr"]),
            max_grad_norm=float(ppm_cfg["max_grad_norm"]),
            vf_coef=float(ppm_cfg["vf_coef"]),
            optimizer_kwargs=dict(ppm_cfg.get("optimizer_kwargs", {})),
        ),
        Candidate(
            label=f"{scope}__proposed_noG",
            method="proposed_noG",
            optimizer="proposed_noG",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs={**proposed_common, "qp_normalization": "block"},
        ),
        Candidate(
            label=f"{scope}__proposed_qp_none",
            method="proposed_qp",
            optimizer="proposed_qp",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs={**proposed_common, "qp_normalization": "none"},
        ),
        Candidate(
            label=f"{scope}__proposed_qp_global",
            method="proposed_qp",
            optimizer="proposed_qp",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs={**proposed_common, "qp_normalization": "global"},
        ),
        Candidate(
            label=f"{scope}__proposed_qp_block",
            method="proposed_qp",
            optimizer="proposed_qp",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs={**proposed_common, "qp_normalization": "block"},
        ),
    ]


def step_candidate(algo, rollout_data, candidate: Candidate) -> Dict[str, object]:
    optimizer_class = get_optimizer_class(candidate.optimizer)
    param_list = optimizer_param_list(algo.policy.optimizer)
    optimizer = optimizer_class(param_list, lr=candidate.lr, **candidate.optimizer_kwargs)
    algo.policy.optimizer = optimizer

    clip_range = algo.clip_range(algo._current_progress_remaining)
    clip_range_vf = None
    if algo.clip_range_vf is not None:
        clip_range_vf = algo.clip_range_vf(algo._current_progress_remaining)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions
    named_params = algo._named_policy_parameters()
    theta_before = clone_named_state(named_params)
    eval_closure = algo._build_eval_closure(rollout_data, actions, clip_range, clip_range_vf)

    old_vf_coef = algo.vf_coef
    old_max_grad_norm = algo.max_grad_norm
    algo.vf_coef = candidate.vf_coef
    algo.max_grad_norm = candidate.max_grad_norm
    try:
        old_eval = eval_closure(theta_override=theta_before, backward=True)
        critic_update_norm_separate = 0.0
        if getattr(optimizer, "requires_eval_closure", False):
            step_loss = optimizer.step(eval_closure=eval_closure, named_params=named_params)
            if algo.optimizer_scope == "actor_game":
                critic_update_norm_separate = float(algo._apply_actor_game_critic_step(eval_closure))
        elif getattr(optimizer, "requires_closure", False):
            def closure():
                return eval_closure(backward=True)["loss_tensor"]
            step_loss = optimizer.step(closure)
            if algo.optimizer_scope == "actor_game":
                critic_update_norm_separate = float(algo._apply_actor_game_critic_step(eval_closure))
        else:
            optimizer.zero_grad()
            closure_info = eval_closure(backward=True)
            step_loss = closure_info["loss_tensor"]
            optimizer.step()
            if algo.optimizer_scope == "actor_game":
                critic_update_norm_separate = float(algo._apply_actor_game_critic_step(eval_closure))

        theta_after = clone_named_state(named_params)
        new_eval = eval_closure(theta_override=theta_after, backward=False)
    finally:
        algo.vf_coef = old_vf_coef
        algo.max_grad_norm = old_max_grad_norm
        for name, param in named_params:
            param.data.copy_(theta_before[name])

    diff = named_difference(theta_after, theta_before)
    selected_names = list(diff.keys())
    metrics = dict(getattr(optimizer, "last_step_metrics", {}))
    metrics.update(
        {
            "scope": algo.optimizer_scope,
            "method": candidate.method,
            "label": candidate.label,
            "optimizer": candidate.optimizer,
            "lr": candidate.lr,
            "max_grad_norm": candidate.max_grad_norm,
            "vf_coef": candidate.vf_coef,
            "qp_normalization": candidate.optimizer_kwargs.get("qp_normalization", ""),
            "qp_g_alpha": candidate.optimizer_kwargs.get("qp_g_alpha", ""),
            "same_minibatch_total_loss_change": float(new_eval["total_loss"] - old_eval["total_loss"]),
            "policy_loss_change": float(new_eval["policy_loss"] - old_eval["policy_loss"]),
            "value_loss_change": float(new_eval["value_loss"] - old_eval["value_loss"]),
            "entropy_change": float(new_eval["entropy_loss"] - old_eval["entropy_loss"]),
            "actor_update_norm": block_norm(diff, selected_names, "actor"),
            "logstd_update_norm": block_norm(diff, selected_names, "logstd"),
            "critic_update_norm": block_norm(diff, selected_names, "critic"),
            "critic_update_norm_separate_adam": critic_update_norm_separate,
            "update_norm_post_cap": float(metrics.get("update_norm_post_cap", float(th.norm(th.cat([tensor.reshape(-1) for tensor in diff.values()])) if diff else 0.0))),
            "optimizer_state_size": len(optimizer.state),
            "solver_status": metrics.get("solver_status", "n/a"),
            "gamma_active_flag": int(float(metrics.get("gamma_active_frac", 0.0)) > 0.0),
            "boundary_solution_flag": int(metrics.get("boundary_solution_flag", 0)),
            "interior_solution_flag": int(metrics.get("interior_solution_flag", 0)),
            "zero_update_flag": int(metrics.get("zero_update_flag", 0)),
        }
    )
    metrics["step_loss"] = float(step_loss.item()) if step_loss is not None else float("nan")
    return metrics


def health_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, label), group in df.groupby(["scope", "label"]):
        rows.append(
            {
                "scope": scope,
                "label": label,
                "method": str(group["method"].iloc[0]),
                "gamma_active_frac_mean": float(group.get("gamma_active_frac", pd.Series([0.0] * len(group))).mean()),
                "loss_change_mean": float(group["same_minibatch_total_loss_change"].mean()),
                "nonfinite_flag": int((~np.isfinite(group["update_norm_post_cap"])).any()),
                "zero_update_frac": float(group["zero_update_flag"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "label"]).reset_index(drop=True)


def plot_scope(df: pd.DataFrame, scope: str, plot_path: pathlib.Path) -> None:
    scope_df = df[df["scope"] == scope].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = scope_df["label"].tolist()
    axes[0].bar(labels, scope_df["loss_change_mean"], color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title(f"{scope}: same-minibatch total loss change")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.3)
    axes[1].bar(labels, scope_df.get("gamma_active_frac_mean", pd.Series([0.0] * len(scope_df))), color="tab:orange")
    axes[1].set_title(f"{scope}: gamma_active_frac mean")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = load_stage5_configs(pathlib.Path(args.stage5_root))
    all_rows: List[Dict[str, object]] = []
    selected_settings: Dict[str, Dict[str, object]] = {}
    health_rows: List[Dict[str, object]] = []

    for scope in ["full_policy", "actor_game"]:
        manager = build_manager(args, scope)
        rarl_model = manager.setup_experiment()
        candidates = make_candidates(configs, scope, args)
        scope_rows: List[Dict[str, object]] = []
        for role in ["protagonist", "adversary"]:
            algo = rarl_model.protagonist if role == "protagonist" else rarl_model.adversary
            probes = collect_probe_batches(rarl_model, role, args.num_probes)
            for probe in probes:
                for candidate in candidates:
                    result = step_candidate(algo, probe.rollout_data, candidate)
                    result["role"] = role
                    result["probe_idx"] = probe.probe_idx
                    scope_rows.append(result)
                    all_rows.append(result)

        scope_df = pd.DataFrame(scope_rows)
        proposed_screen = scope_df[scope_df["method"] == "proposed_qp"].copy()
        score_rows = []
        for label, group in proposed_screen.groupby("label"):
            score_rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "qp_normalization": str(group["qp_normalization"].iloc[0]),
                    "qp_g_alpha": float(group["qp_g_alpha"].iloc[0]),
                    "gamma_active_frac_mean": float(group.get("gamma_active_frac", pd.Series([0.0] * len(group))).mean()),
                    "loss_change_mean": float(group["same_minibatch_total_loss_change"].mean()),
                    "zero_update_frac": float(group["zero_update_flag"].mean()),
                    "score": 5.0 * float(group.get("gamma_active_frac", pd.Series([0.0] * len(group))).mean())
                    - 3.0 * max(float(group["same_minibatch_total_loss_change"].mean()), 0.0)
                    - 2.0 * float(group["zero_update_flag"].mean()),
                }
            )
        score_df = pd.DataFrame(score_rows).sort_values("score", ascending=False).reset_index(drop=True)
        winner = score_df.iloc[0]
        selected_settings[scope] = {
            "qp_normalization": winner["qp_normalization"],
            "qp_g_alpha": float(winner["qp_g_alpha"]),
            "label": winner["label"],
        }
        health_rows.extend(score_rows)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output_root / "stage8c_fixed_minibatch_audit.csv", index=False)
    health_df = health_summary(all_df)
    health_df.to_csv(output_root / "stage8c_health_summary.csv", index=False)
    (output_root / "stage8_selected_qp_settings.json").write_text(json.dumps(selected_settings, indent=2), encoding="utf-8")

    for scope in ["full_policy", "actor_game"]:
        plot_scope(
            health_df[health_df["scope"] == scope],
            scope,
            plots_dir / f"stage8c_fixed_minibatch_audit_{scope}.png",
        )

    lines = [
        "# Stage 8C Fixed-Minibatch Audit",
        "",
        f"- Environment: `{args.env}`",
        f"- Seed: `{args.seed}`",
        f"- Probes per role: `{args.num_probes}`",
        "",
    ]
    gate_lines = []
    for scope in ["full_policy", "actor_game"]:
        scope_df = all_df[(all_df["scope"] == scope) & (all_df["method"] == "proposed_qp") & (all_df["qp_normalization"] == selected_settings[scope]["qp_normalization"])]
        no_nan = bool(np.isfinite(scope_df["update_norm_post_cap"]).all())
        nonzero = bool((scope_df["update_norm_post_cap"] > 0).all())
        gamma_active = float(scope_df.get("gamma_active_frac", pd.Series([0.0] * len(scope_df))).mean()) > 0.0
        loss_not_worse = float(scope_df["same_minibatch_total_loss_change"].mean()) <= 0.0
        noG_df = all_df[(all_df["scope"] == scope) & (all_df["method"] == "proposed_noG")]
        not_worse_than_noG = float(scope_df["same_minibatch_total_loss_change"].mean()) <= float(noG_df["same_minibatch_total_loss_change"].mean()) + 0.01
        gate = no_nan and nonzero and gamma_active and loss_not_worse and not_worse_than_noG
        gate_lines.extend(
            [
                f"## {scope}",
                "",
                f"- selected proposed_qp normalization: `{selected_settings[scope]['qp_normalization']}`",
                f"- proposed_qp update nonzero: `{nonzero}`",
                f"- proposed_qp gamma_active_frac mean > 0: `{gamma_active}`",
                f"- proposed_qp same-minibatch total loss not systematically worse: `{loss_not_worse}`",
                f"- proposed_qp not obviously worse than proposed_noG: `{not_worse_than_noG}`",
                f"- scope audit gate pass: `{gate}`",
                "",
            ]
        )
    lines.extend(gate_lines)
    (output_root / "stage8c_fixed_minibatch_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
