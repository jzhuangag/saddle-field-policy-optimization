from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("gda", "egm", "ppm", "nog", "qpg")
COLORS = {"gda": "#444444", "egm": "#E69F00", "ppm": "#009E73", "nog": "#2878B5", "qpg": "#C53D32"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    curves, updates = [], []
    for method in METHODS:
        c = pd.read_csv(args.root / method / "convergence.csv")
        u = pd.read_csv(args.root / method / "updates.csv")
        c["method"] = method
        u["method"] = method
        curves.append(c)
        updates.append(u)
    curves = pd.concat(curves, ignore_index=True)
    updates = pd.concat(updates, ignore_index=True)
    curves.to_csv(args.output / "all_convergence.csv", index=False)
    updates.to_csv(args.output / "all_updates.csv", index=False)

    finals = curves.sort_values("step").groupby("method", as_index=False).tail(1).set_index("method")
    summary = {"runs_complete": int(len(finals)), "methods": {}}
    for method in METHODS:
        row = finals.loc[method]
        u = updates[updates.method == method]
        item = {
            "final_collection_return": float(row.train_episode_return),
            "final_clean_return": float(row.clean_return),
            "final_own_robust_return": float(row.own_adversary_robust_return),
            "final_corr_Q_MC": float(row.corr_Q_MC),
            "final_WF_over_SF": float(row.WF_over_SF),
            "max_WF_over_SF": float(curves[curves.method == method].WF_over_SF.max()),
            "skew_dominant_checkpoint_fraction": float(curves[curves.method == method].skew_dominant_along_F.mean()),
            "fallback_fraction": float(u.fallback.mean()),
        }
        if "gamma_active" in u:
            item["gamma_active_fraction"] = float(u.gamma_active.mean())
        if "realized_field_energy_change" in u:
            item["field_energy_decrease_fraction"] = float((u.realized_field_energy_change.fillna(np.inf) <= 0).mean())
            item["G_contribution_norm_mean"] = float(u.G_contribution_norm.fillna(0).mean())
        summary["methods"][method] = item
    summary["qpg_minus_nog_final_robust"] = summary["methods"]["qpg"]["final_own_robust_return"] - summary["methods"]["nog"]["final_own_robust_return"]
    summary["qpg_minus_nog_final_clean"] = summary["methods"]["qpg"]["final_clean_return"] - summary["methods"]["nog"]["final_clean_return"]
    summary["locomotion_gate"] = bool(min(v["final_clean_return"] for v in summary["methods"].values()) >= 1000)
    summary["rotation_gate"] = bool(summary["methods"]["qpg"]["max_WF_over_SF"] >= 1.0)
    summary["single_seed_positive_gate"] = bool(summary["qpg_minus_nog_final_robust"] > 0)
    summary["decision"] = "PASS" if summary["locomotion_gate"] and summary["rotation_gate"] and summary["single_seed_positive_gate"] else "FAIL"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("train_episode_return", "Protagonist collection return", "Native task return"),
        ("own_adversary_robust_return", "Co-trained adversary robust evaluation", "Native task return"),
        ("clean_return", "Clean evaluation", "Native task return"),
        ("G_norm", "Joint curvature norm", "||J_F F||"),
        ("WF_over_SF", "Directional rotation ratio", "||WF|| / ||SF||"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat[:5], panels):
        for method in METHODS:
            data = curves[curves.method == method].sort_values("step")
            ax.plot(data.step / 1000.0, data[metric], label=method, color=COLORS[method], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Joint-training environment steps (thousands)")
        ax.set_ylabel(ylabel)
        if metric == "WF_over_SF":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="skew threshold")
        ax.legend(fontsize=8)
    ax = axes.flat[5]
    no = curves[curves.method == "nog"].sort_values("step")
    qp = curves[curves.method == "qpg"].sort_values("step")
    ax.plot(qp.step / 1000.0, qp.own_adversary_robust_return.to_numpy() - no.own_adversary_robust_return.to_numpy(), color=COLORS["qpg"], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("QP+G minus noG co-trained robust return")
    ax.set_xlabel("Joint-training environment steps (thousands)")
    ax.set_ylabel("Return difference")
    fig.savefig(args.output / "radial_warm_joint_gate_convergence.png", dpi=180)
    fig.savefig(args.output / "radial_warm_joint_gate_convergence.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
