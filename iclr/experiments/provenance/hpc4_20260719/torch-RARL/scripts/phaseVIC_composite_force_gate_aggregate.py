from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


FORCES = (0.5, 1.0, 2.5)
PRETRAIN_CLEAN_RETURN = 1837.79


def finite_mean(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else math.nan


def fraction(values: pd.Series, predicate) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(predicate(x))) if x.size else math.nan


def summarize(root: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for force in FORCES:
        run = root / "HalfCheetah-v4" / f"force_{force}" / "seed_0" / "nog"
        conv_path = run / "convergence.csv"
        updates_path = run / "updates.csv"
        if not conv_path.exists() or not updates_path.exists():
            raise FileNotFoundError(f"Incomplete gate run: {run}")
        conv = pd.read_csv(conv_path)
        updates = pd.read_csv(updates_path)
        final = conv.sort_values("step").iloc[-1]
        gain = pd.to_numeric(updates["counterfactual_qpg_gain"], errors="coerce")
        gain_scale = pd.to_numeric(updates["predicted_nog_change"], errors="coerce").abs().clip(lower=1e-12)
        normalized_gain = gain / gain_scale
        row = {
            "force_max": force,
            "updates": int(len(updates)),
            "final_clean_return": float(final["clean_return"]),
            "final_robust_return": float(final["own_adversary_robust_return"]),
            "clean_retention": float(final["clean_return"]) / PRETRAIN_CLEAN_RETURN,
            "final_corr_Q_MC": float(final["corr_Q_MC"]),
            "corr_gate_checkpoint_fraction": fraction(conv["corr_Q_MC"], lambda x: x >= 0.60),
            "critic_gate_update_fraction": finite_mean(updates["curvature_reliability_gate"]),
            "mean_WF_over_SF": finite_mean(conv["WF_over_SF"]),
            "max_WF_over_SF": float(pd.to_numeric(conv["WF_over_SF"], errors="coerce").max()),
            "skew_checkpoint_fraction": finite_mean(conv["skew_dominant_along_F"]),
            "counterfactual_G_active_fraction": fraction(updates["counterfactual_qpg_G_norm"], lambda x: x > 1e-10),
            "counterfactual_gain_positive_fraction": fraction(gain, lambda x: x > 1e-10),
            "counterfactual_gain_mean": finite_mean(gain),
            "counterfactual_normalized_gain_mean": finite_mean(normalized_gain),
            "predicted_inclusion_pass_fraction": finite_mean(updates["predicted_inclusion_pass"]),
            "realized_nog_safeguard_pass_fraction": finite_mean(updates["realized_qp_le_nog"]),
            "zero_update_fraction": fraction(
                pd.to_numeric(updates["beta"], errors="coerce").abs()
                + pd.to_numeric(updates["gamma"], errors="coerce").abs(),
                lambda x: x <= 1e-14,
            ),
            "backtrack_fraction": fraction(updates["backtracks"], lambda x: x > 0),
            "median_pd_inflation": float(pd.to_numeric(updates["qp_pd_inflation"], errors="coerce").median()),
        }
        row["learning_gate"] = bool(row["final_clean_return"] >= 1000 and row["clean_retention"] >= 0.70)
        row["critic_gate"] = bool(
            row["corr_gate_checkpoint_fraction"] >= 0.60
            and row["critic_gate_update_fraction"] >= 0.60
        )
        row["numerical_gate"] = bool(
            row["predicted_inclusion_pass_fraction"] == 1.0
            and row["realized_nog_safeguard_pass_fraction"] == 1.0
            and row["zero_update_fraction"] <= 0.30
        )
        row["rotation_gate"] = bool(
            row["counterfactual_G_active_fraction"] >= 0.10
            and row["counterfactual_gain_positive_fraction"] >= 0.10
            and row["counterfactual_normalized_gain_mean"] > 0.0
        )
        row["all_gates"] = bool(row["learning_gate"] and row["critic_gate"] and row["numerical_gate"] and row["rotation_gate"])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("force_max").reset_index(drop=True)
    eligible = summary.loc[summary["all_gates"]]
    selected = float(eligible.iloc[0]["force_max"]) if not eligible.empty else None
    decision = {
        "selection_rule": "smallest radial force satisfying pre-registered learning, critic, numerical, and reduced-gradient gates",
        "selection_uses_qpg_performance": False,
        "selected_force_max": selected,
        "decision": "PASS" if selected is not None else "NO_HALFCHEETAH_FORCE_PASSES",
    }
    return summary, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, decision = summarize(args.root)
    summary.to_csv(args.output / "force_gate_summary.csv", index=False)
    (args.output / "force_gate_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
