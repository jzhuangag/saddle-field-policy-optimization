from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sem(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    curves, updates = [], []
    for seed in range(5):
        for method in ("nog", "qpg"):
            run = args.root / f"seed_{seed}" / method
            c = pd.read_csv(run / "convergence.csv")
            u = pd.read_csv(run / "updates.csv")
            c["seed"] = seed
            c["method"] = method
            u["seed"] = seed
            u["method"] = method
            curves.append(c)
            updates.append(u)
    curves_df = pd.concat(curves, ignore_index=True)
    updates_df = pd.concat(updates, ignore_index=True)
    curves_df.to_csv(args.output / "all_convergence.csv", index=False)
    updates_df.to_csv(args.output / "all_updates.csv", index=False)

    finals = curves_df.sort_values("step").groupby(["seed", "method"], as_index=False).tail(1)
    paired_rows = []
    for seed in range(5):
        no = finals[(finals.seed == seed) & (finals.method == "nog")].iloc[0]
        qp = finals[(finals.seed == seed) & (finals.method == "qpg")].iloc[0]
        paired_rows.append({
            "seed": seed,
            "clean_qpg_minus_nog": qp.clean_return - no.clean_return,
            "robust_qpg_minus_nog": qp.own_adversary_robust_return - no.own_adversary_robust_return,
            "qpg_clean": qp.clean_return,
            "nog_clean": no.clean_return,
            "qpg_robust": qp.own_adversary_robust_return,
            "nog_robust": no.own_adversary_robust_return,
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output / "paired_final.csv", index=False)

    qpg_curves = curves_df[curves_df.method == "qpg"]
    qpg_updates = updates_df[updates_df.method == "qpg"]
    diagnostics = {
        "runs_complete": int(finals.shape[0]),
        "final_step_min": int(finals.step.min()),
        "critic_corr_Q_MC_all_mean": float(qpg_curves.corr_Q_MC.mean()),
        "critic_corr_Q_MC_final_mean": float(finals[finals.method == "qpg"].corr_Q_MC.mean()),
        "WF_over_SF_all_mean": float(qpg_curves.WF_over_SF.mean()),
        "WF_over_SF_all_max": float(qpg_curves.WF_over_SF.max()),
        "WF_over_SF_final_mean": float(finals[finals.method == "qpg"].WF_over_SF.mean()),
        "skew_dominant_checkpoint_fraction": float(qpg_curves.skew_dominant_along_F.mean()),
        "cross_player_coupling_all_mean": float(qpg_curves.cross_player_coupling_proxy.mean()),
        "gamma_active_fraction": float(qpg_updates.gamma_active.mean()),
        "fallback_fraction": float(qpg_updates.fallback.mean()),
        "realized_field_energy_decrease_fraction": float((qpg_updates.realized_field_energy_change <= 0).mean()),
        "realized_field_energy_change_mean": float(qpg_updates.realized_field_energy_change.mean()),
        "G_contribution_norm_mean": float(qpg_updates.G_contribution_norm.mean()),
        "robust_gain_mean": float(paired.robust_qpg_minus_nog.mean()),
        "robust_gain_sem": sem(paired.robust_qpg_minus_nog.to_numpy()),
        "robust_seed_wins": int((paired.robust_qpg_minus_nog > 0).sum()),
        "clean_gain_mean": float(paired.clean_qpg_minus_nog.mean()),
        "clean_gain_sem": sem(paired.clean_qpg_minus_nog.to_numpy()),
        "clean_seed_wins": int((paired.clean_qpg_minus_nog > 0).sum()),
        "final_qpg_robust_mean": float(paired.qpg_robust.mean()),
        "final_nog_robust_mean": float(paired.nog_robust.mean()),
        "final_qpg_clean_mean": float(paired.qpg_clean.mean()),
        "final_nog_clean_mean": float(paired.nog_clean.mean()),
    }
    diagnostics["critic_gate"] = bool(diagnostics["critic_corr_Q_MC_final_mean"] >= 0.5)
    diagnostics["rotation_gate"] = bool(
        diagnostics["WF_over_SF_all_max"] >= 1.0
        and diagnostics["skew_dominant_checkpoint_fraction"] >= 0.1
        and diagnostics["gamma_active_fraction"] >= 0.05
    )
    diagnostics["learning_gate"] = bool(max(diagnostics["final_qpg_clean_mean"], diagnostics["final_nog_clean_mean"]) >= 1000.0)
    diagnostics["performance_gate"] = bool(diagnostics["robust_gain_mean"] > 0 and diagnostics["robust_seed_wins"] >= 3)
    diagnostics["decision"] = "PASS" if all(
        diagnostics[k] for k in ("critic_gate", "rotation_gate", "learning_gate", "performance_gate")
    ) else "FAIL"
    (args.output / "gate_summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("train_episode_return", "Protagonist collection return"),
        ("own_adversary_robust_return", "Co-trained adversary robust evaluation"),
        ("clean_return", "Clean evaluation"),
    ]
    colors = {"nog": "#2878B5", "qpg": "#C53D32"}
    for ax, (metric, title) in zip(axes.flat[:3], panels):
        for method in ("nog", "qpg"):
            table = curves_df[curves_df.method == method].pivot(index="step", columns="seed", values=metric)
            mean = table.mean(axis=1)
            err = table.sem(axis=1)
            ax.plot(mean.index / 1000.0, mean, label=method, color=colors[method])
            ax.fill_between(mean.index / 1000.0, mean - err, mean + err, color=colors[method], alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("Adversarial-phase environment steps (thousands)")
        ax.set_ylabel("Native task return")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.legend()
    gain_ax = axes.flat[3]
    no = curves_df[curves_df.method == "nog"].pivot(index="step", columns="seed", values="own_adversary_robust_return")
    qp = curves_df[curves_df.method == "qpg"].pivot(index="step", columns="seed", values="own_adversary_robust_return")
    gain = qp - no
    gain_ax.plot(gain.index / 1000.0, gain.mean(axis=1), color=colors["qpg"])
    gain_ax.fill_between(gain.index / 1000.0, gain.mean(axis=1) - gain.sem(axis=1), gain.mean(axis=1) + gain.sem(axis=1), color=colors["qpg"], alpha=0.18)
    gain_ax.axhline(0, color="black", linewidth=0.8)
    gain_ax.set_title("Paired robust gain: QP+G minus noG")
    gain_ax.set_xlabel("Adversarial-phase environment steps (thousands)")
    gain_ax.set_ylabel("Native task return difference")
    fig.savefig(args.output / "exact_joint_gate_convergence.png", dpi=180)
    fig.savefig(args.output / "exact_joint_gate_convergence.pdf")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
