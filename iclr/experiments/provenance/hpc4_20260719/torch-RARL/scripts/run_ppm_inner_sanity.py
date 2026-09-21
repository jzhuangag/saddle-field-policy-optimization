from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.optimizers import PPM, clone_named_state, named_tensor_norm, restore_named_state
from scripts.full_policy_optimizer_probe import (
    build_manager,
    collect_probe_batches,
    compute_clip_ranges,
    compute_loss_and_grads,
    cosine_similarity,
    egm_update,
    named_parameters,
    ppm_update,
    state_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 7A PPM inner-loop sanity closeout")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes", type=int, default=3)
    return parser.parse_args()


def manual_closure_factory(algo, rollout_data):
    clip_range, clip_range_vf = compute_clip_ranges(algo)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions
    counter = {"calls": 0}

    def closure():
        counter["calls"] += 1
        algo.policy.optimizer.zero_grad()
        total_loss, *_ = algo._build_shared_policy_loss(rollout_data, actions, clip_range, clip_range_vf)
        total_loss.backward()
        if np.isfinite(algo.max_grad_norm):
            th.nn.utils.clip_grad_norm_(algo.policy.parameters(), algo.max_grad_norm)
        return total_loss

    return closure, counter


def actual_ppm_step_check(algo, rollout_data, inner_steps: int) -> Dict[str, object]:
    named_params = named_parameters(algo.policy)
    theta_old = clone_named_state(named_params)
    optimizer = PPM(algo.policy.parameters(), lr=float(algo.learning_rate if isinstance(algo.learning_rate, float) else algo.lr_schedule(1.0)), inner_steps=inner_steps)
    algo.policy.optimizer = optimizer
    closure, counter = manual_closure_factory(algo, rollout_data)
    state_size_before = len(optimizer.state)
    optimizer.step(closure)
    state_size_after = len(optimizer.state)
    theta_new = clone_named_state(named_params)
    restore_named_state(named_params, theta_old)
    return {
        "closure_calls": counter["calls"],
        "state_size_before": state_size_before,
        "state_size_after": state_size_after,
        "update_norm": named_tensor_norm({name: theta_new[name] - theta_old[name] for name in theta_old}),
    }


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    manager = build_manager(args)
    rarl_model = manager.setup_experiment()

    ppm_step_source = inspect.getsource(PPM.step)
    source_uses_optimizer_step = "optimizer.step(" in ppm_step_source

    rows: List[Dict[str, object]] = []
    summary_lines: List[str] = [
        "# PPM Inner Sanity Report",
        "",
        f"- Environment: `{args.env}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{args.device}`",
        f"- Number of probes per role: `{args.num_probes}`",
        "",
        "## Static source checks",
        "",
        f"- `PPM.step()` contains `optimizer.step(...)`: `{source_uses_optimizer_step}`",
    ]

    for role, algo in [
        ("protagonist", rarl_model.protagonist),
        ("adversary", rarl_model.adversary),
    ]:
        probes = collect_probe_batches(rarl_model, role, args.num_probes)
        default_lr = float(algo.learning_rate if isinstance(algo.learning_rate, float) else algo.lr_schedule(1.0))
        default_vf_coef = float(algo.vf_coef)
        default_ent_coef = float(algo.ent_coef)
        default_max_grad_norm = float(algo.max_grad_norm)

        for probe in probes:
            named_params = named_parameters(algo.policy)
            theta_old = clone_named_state(named_params)
            _, _, _, theta_egm = egm_update(
                algo,
                probe.rollout_data,
                theta_old,
                default_lr,
                max_grad_norm=default_max_grad_norm,
                vf_coef=default_vf_coef,
                ent_coef=default_ent_coef,
            )

            for inner_steps in [2, 5, 10, 20]:
                actual_step = actual_ppm_step_check(algo, probe.rollout_data, inner_steps)
                last_eval, theta_ppm, inner_losses, inner_policy_losses, inner_value_losses, inner_entropy_losses, _, inner_residuals = ppm_update(
                    algo,
                    probe.rollout_data,
                    theta_old,
                    default_lr,
                    inner_steps=inner_steps,
                    max_grad_norm=default_max_grad_norm,
                    vf_coef=default_vf_coef,
                    ent_coef=default_ent_coef,
                )
                grad_residuals: List[float] = []
                # Recompute grad residuals along the exact fixed-point path.
                prev_grads = None
                theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
                for _ in range(inner_steps):
                    restore_named_state(named_params, theta_tmp)
                    eval_info = compute_loss_and_grads(
                        algo,
                        probe.rollout_data,
                        max_grad_norm=default_max_grad_norm,
                        vf_coef=default_vf_coef,
                        ent_coef=default_ent_coef,
                    )
                    if prev_grads is not None:
                        grad_residuals.append(
                            named_tensor_norm({name: eval_info["grads"][name] - prev_grads[name] for name in prev_grads})
                        )
                    theta_tmp = {name: theta_old[name] - default_lr * eval_info["grads"][name] for name in theta_old}
                    prev_grads = eval_info["grads"]
                restore_named_state(named_params, theta_old)

                theta_diff_vs_egm = named_tensor_norm({name: theta_ppm[name] - theta_egm[name] for name in theta_old})
                row = {
                    "role": role,
                    "probe_idx": probe.probe_idx,
                    "inner_steps": inner_steps,
                    "lr": default_lr,
                    "max_grad_norm": default_max_grad_norm,
                    "vf_coef": default_vf_coef,
                    "closure_calls_actual_step": actual_step["closure_calls"],
                    "state_size_before": actual_step["state_size_before"],
                    "state_size_after": actual_step["state_size_after"],
                    "actual_step_update_norm": actual_step["update_norm"],
                    "theta_diff_vs_egm": theta_diff_vs_egm,
                    "update_cosine_vs_egm": cosine_similarity(
                        state_vector({name: theta_ppm[name] - theta_old[name] for name in theta_old}),
                        state_vector({name: theta_egm[name] - theta_old[name] for name in theta_old}),
                    ),
                    "inner_theta_residuals": json.dumps(inner_residuals),
                    "inner_grad_residuals": json.dumps(grad_residuals),
                    "inner_total_losses": json.dumps(inner_losses),
                    "inner_policy_losses": json.dumps(inner_policy_losses),
                    "inner_value_losses": json.dumps(inner_value_losses),
                    "inner_entropy_losses": json.dumps(inner_entropy_losses),
                    "inner_total_loss_last_minus_first": float(inner_losses[-1] - inner_losses[0]),
                    "inner_theta_residual_first": float(inner_residuals[0]) if inner_residuals else float("nan"),
                    "inner_theta_residual_last": float(inner_residuals[-1]) if inner_residuals else float("nan"),
                    "inner_grad_residual_first": float(grad_residuals[0]) if grad_residuals else float("nan"),
                    "inner_grad_residual_last": float(grad_residuals[-1]) if grad_residuals else float("nan"),
                    "final_total_loss": float(last_eval["total_loss"]),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_root / "ppm_inner_sanity.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for inner_steps, group in df.groupby("inner_steps"):
        theta_first = group["inner_theta_residual_first"].mean()
        theta_last = group["inner_theta_residual_last"].mean()
        grad_first = group["inner_grad_residual_first"].mean()
        grad_last = group["inner_grad_residual_last"].mean()
        axes[0].plot([1, inner_steps], [theta_first, theta_last], marker="o", label=f"inner={inner_steps}")
        axes[1].plot([2, inner_steps], [grad_first, grad_last], marker="o", label=f"inner={inner_steps}")
    axes[0].set_title("PPM Theta Residual by Steps")
    axes[0].set_xlabel("Inner-step index (first vs last)")
    axes[0].set_ylabel("||theta_m - theta_(m-1)||")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("PPM Gradient Residual by Steps")
    axes[1].set_xlabel("Inner-step index (first available vs last)")
    axes[1].set_ylabel("||F(theta_m) - F(theta_(m-1))||")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ppm_inner_residual_by_steps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    inner2 = df[df["inner_steps"] == 2]
    inner20 = df[df["inner_steps"] == 20]
    summary_lines.extend(
        [
            "",
            "## Probe conclusions",
            "",
            f"- inner=2 mean theta_diff_vs_egm: `{inner2['theta_diff_vs_egm'].mean():.12f}`",
            f"- inner=2 mean update cosine vs EGM: `{inner2['update_cosine_vs_egm'].mean():.12f}`",
            f"- inner=20 mean theta residual first/last: `{inner20['inner_theta_residual_first'].mean():.6e}` / `{inner20['inner_theta_residual_last'].mean():.6e}`",
            f"- inner=20 mean grad residual first/last: `{inner20['inner_grad_residual_first'].mean():.6e}` / `{inner20['inner_grad_residual_last'].mean():.6e}`",
            f"- Actual PPM.step closure calls for inner=20: `{int(inner20['closure_calls_actual_step'].mode().iloc[0])}`",
            f"- Any optimizer state created by PPM: `{bool((df['state_size_after'] > 0).any())}`",
            "",
            "## Interpretation",
            "",
            "- inner=2 matching EGM is expected for the current fixed-point form.",
            "- inner=20 uses the same minibatch repeatedly, so any degradation relative to EGM can still come from stale-minibatch PPO clipping or local loss-shape effects, not from Adam-state contamination.",
            "- No Adam moment/state should appear because PPM is a pure manual fixed-point update and its optimizer.state stays empty in the actual step check.",
        ]
    )
    (output_root / "ppm_inner_sanity_report.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
