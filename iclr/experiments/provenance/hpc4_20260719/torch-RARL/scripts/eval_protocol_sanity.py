from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import load_rarl_for_eval, load_stage5_best_runs, set_rarl_eval_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 6A eval protocol sanity")
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    return parser.parse_args()


def evaluate_strength(model, vec_env, *, strength: float, n_eval_episodes: int) -> Dict[str, float]:
    set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=strength)
    episode_returns: List[float] = []
    applied_force_norms: List[float] = []
    applied_disturbance_norms: List[float] = []
    adv_pre_norms: List[float] = []
    adv_post_norms: List[float] = []
    state_norms: List[float] = []

    for _ in range(n_eval_episodes):
        obs = vec_env.reset()
        done = np.array([False])
        ep_return = 0.0
        step_force_norms = []
        step_disturbance_norms = []
        step_adv_pre = []
        step_adv_post = []
        step_state_norms = []

        while not bool(done[0]):
            action, _ = model.protagonist.predict(obs, deterministic=True)
            obs, reward, done, infos = vec_env.step(action)
            ep_return += float(reward[0])
            info = infos[0]
            step_force_norms.append(float(info.get("applied_force_norm", 0.0)))
            step_disturbance_norms.append(float(info.get("applied_disturbance_norm", 0.0)))
            step_adv_pre.append(float(info.get("adversary_action_norm_pre_clip", 0.0)))
            step_adv_post.append(float(info.get("adversary_action_norm_post_clip", 0.0)))
            step_state_norms.append(float(info.get("state_norm", 0.0)))

        episode_returns.append(ep_return)
        applied_force_norms.append(float(np.mean(step_force_norms)) if step_force_norms else 0.0)
        applied_disturbance_norms.append(float(np.mean(step_disturbance_norms)) if step_disturbance_norms else 0.0)
        adv_pre_norms.append(float(np.mean(step_adv_pre)) if step_adv_pre else 0.0)
        adv_post_norms.append(float(np.mean(step_adv_post)) if step_adv_post else 0.0)
        state_norms.append(float(np.mean(step_state_norms)) if step_state_norms else 0.0)

    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "applied_force_norm": float(np.mean(applied_force_norms)),
        "applied_disturbance_norm": float(np.mean(applied_disturbance_norms)),
        "adversary_action_norm_pre_clip": float(np.mean(adv_pre_norms)),
        "adversary_action_norm_post_clip": float(np.mean(adv_post_norms)),
        "state_norm": float(np.mean(state_norms)),
    }


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    stage5_runs = load_stage5_best_runs(pathlib.Path(args.stage5_root))
    strengths = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []

    for method, saved_run in stage5_runs.items():
        for adv_impact in ["force", "control"]:
            model, vec_env = load_rarl_for_eval(saved_run, adv_impact=adv_impact, adv_strength=1.0, device=args.device)
            try:
                for strength in strengths:
                    stats = evaluate_strength(model, vec_env, strength=strength, n_eval_episodes=args.n_eval_episodes)
                    rows.append(
                        {
                            "method": method,
                            "adv_impact": adv_impact,
                            "adv_strength": strength,
                            "eval_kind": "clean" if strength == 0.0 else "adversarial",
                            **stats,
                        }
                    )
            finally:
                vec_env.close()

    df = pd.DataFrame(rows)
    csv_path = output_root / "eval_protocol_sanity.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, adv_impact in zip(axes, ["force", "control"]):
        impact_df = df[df["adv_impact"] == adv_impact]
        for method, group in impact_df.groupby("method"):
            axis.errorbar(group["adv_strength"], group["mean_return"], yerr=group["std_return"], marker="o", linewidth=1.2, capsize=3, label=method)
        axis.set_title(f"{adv_impact.capitalize()} strength sweep")
        axis.set_xlabel("Adversary strength")
        axis.set_ylabel("Mean return")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "eval_strength_sweep_force_vs_control.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Eval Protocol Sanity",
        "",
        f"- Stage5 source: `{args.stage5_root}`",
        f"- Deterministic eval: `True`",
        f"- Episodes per point: `{args.n_eval_episodes}`",
        "",
        "## Checks",
        "",
    ]
    for method in sorted(df["method"].unique()):
        method_df = df[df["method"] == method]
        for adv_impact in ["force", "control"]:
            slice_df = method_df[method_df["adv_impact"] == adv_impact].sort_values("adv_strength")
            clean_row = slice_df[slice_df["adv_strength"] == 0.0].iloc[0]
            strongest_row = slice_df[slice_df["adv_strength"] == slice_df["adv_strength"].max()].iloc[0]
            lines.extend(
                [
                    f"- `{method}` / `{adv_impact}`",
                    f"  - clean applied_disturbance_norm: `{clean_row['applied_disturbance_norm']:.6f}`",
                    f"  - strongest applied_disturbance_norm: `{strongest_row['applied_disturbance_norm']:.6f}`",
                    f"  - clean mean_return: `{clean_row['mean_return']:.6f}`",
                    f"  - strongest mean_return: `{strongest_row['mean_return']:.6f}`",
                ]
            )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- `control` uses a dimension-preserving embedding of the force-trained adversary output into the first action coordinates for eval-only sensitivity checking.",
            "- This makes the force/control comparison a protocol sanity check, not a claim that the same adversary was trained under both impact models.",
        ]
    )
    (output_root / "eval_protocol_sanity_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
