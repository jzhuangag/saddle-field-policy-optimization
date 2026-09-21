from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = "last5_adversarial_mean"
CLEAN = "last5_clean_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate the pre-registered VI-C HalfCheetah multi-seed confirmation")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def paired_bootstrap(values: np.ndarray, seed: int = 20260718, draws: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def read_seed(root: Path, seed: int) -> pd.DataFrame:
    path = root / f"seed_{seed}" / "stage7e_short_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["confirmation_seed"] = seed
    return frame


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = pd.concat([read_seed(input_root, seed) for seed in args.seeds], ignore_index=True)
    all_rows.to_csv(output_root / "phaseVIC_all_seed_summaries.csv", index=False)

    paired_rows: list[dict[str, float | int]] = []
    for seed in args.seeds:
        frame = all_rows[all_rows["confirmation_seed"] == seed].set_index("method")
        qp = frame.loc["proposed_qp"]
        nog = frame.loc["proposed_noG"]
        paired_rows.append(
            {
                "seed": seed,
                "qp_force_return": float(qp[PRIMARY]),
                "nog_force_return": float(nog[PRIMARY]),
                "force_gain": float(qp[PRIMARY] - nog[PRIMARY]),
                "qp_clean_return": float(qp[CLEAN]),
                "nog_clean_return": float(nog[CLEAN]),
                "clean_gain": float(qp[CLEAN] - nog[CLEAN]),
                "qp_crash": int(qp["crash_flag"]),
                "qp_nan": int(qp["nan_flag"]),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_root / "phaseVIC_paired_qp_vs_nog.csv", index=False)

    diagnostics = []
    for seed in args.seeds:
        path = input_root / f"seed_{seed}" / "proposed_qp_diagnostics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["confirmation_seed"] = seed
            diagnostics.append(frame)
    diag = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    if not diag.empty:
        diag.to_csv(output_root / "phaseVIC_qp_diagnostics_all_seeds.csv", index=False)
    qp_diag = diag[diag["method"] == "proposed_qp"].copy() if "method" in diag else diag

    force = paired["force_gain"].to_numpy(dtype=np.float64)
    clean = paired["clean_gain"].to_numpy(dtype=np.float64)
    force_ci = paired_bootstrap(force)
    clean_ci = paired_bootstrap(clean)
    force_wins = int((force > 0).sum())
    clean_wins = int((clean > 0).sum())
    mean_nog_clean = float(paired["nog_clean_return"].mean())
    clean_noninferior = float(paired["qp_clean_return"].mean()) >= 0.9 * mean_nog_clean
    healthy = int(paired[["qp_crash", "qp_nan"]].to_numpy().sum()) == 0
    gamma_active = float((qp_diag["gamma"].abs() > 1e-12).mean()) if "gamma" in qp_diag else float("nan")

    # Pre-registered decision: majority paired robust wins, positive mean robust gain,
    # no more than 10% clean-return loss, active G, and no crashes/NaNs.
    gate = bool(force_wins >= 3 and force.mean() > 0 and clean_noninferior and gamma_active >= 0.1 and healthy)
    decision = "POSITIVE_MULTISEED" if gate else "NOT_YET_POSITIVE"
    summary = {
        "decision": decision,
        "seeds": args.seeds,
        "force_wins_qp_over_nog": force_wins,
        "clean_wins_qp_over_nog": clean_wins,
        "mean_force_gain": float(force.mean()),
        "median_force_gain": float(np.median(force)),
        "force_gain_bootstrap_95ci": list(force_ci),
        "mean_clean_gain": float(clean.mean()),
        "clean_gain_bootstrap_95ci": list(clean_ci),
        "clean_noninferior_10pct": clean_noninferior,
        "gamma_active_fraction": gamma_active,
        "healthy_all_seeds": healthy,
    }
    (output_root / "phaseVIC_multiseed_decision.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# VI-C HalfCheetah PPO-RARL Multi-seed Confirmation",
        "",
        "Pre-registered primary endpoint: paired QP+G minus noG last-five force-adversary evaluation return.",
        "Native Gymnasium reward is retained; the adversary applies the standard two-dimensional MuJoCo torso force.",
        "",
        f"- decision: `{decision}`",
        f"- force-return wins: `{force_wins}/{len(args.seeds)}`",
        f"- mean paired force gain: `{force.mean():.6f}`",
        f"- paired force-gain bootstrap 95% CI: `[{force_ci[0]:.6f}, {force_ci[1]:.6f}]`",
        f"- clean-return wins: `{clean_wins}/{len(args.seeds)}`",
        f"- mean paired clean gain: `{clean.mean():.6f}`",
        f"- clean non-inferiority at 10%: `{clean_noninferior}`",
        f"- gamma-active fraction: `{gamma_active:.6f}`",
        f"- all QP+G runs finite and crash-free: `{healthy}`",
    ]
    (output_root / "phaseVIC_multiseed_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    x = np.arange(len(args.seeds))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, qp_col, nog_col, title in [
        (axes[0], "qp_force_return", "nog_force_return", "Force-adversary return"),
        (axes[1], "qp_clean_return", "nog_clean_return", "Clean return"),
    ]:
        axis.bar(x - width / 2, paired[nog_col], width, label="noG", color="#7a8793")
        axis.bar(x + width / 2, paired[qp_col], width, label="QP+G", color="#c84b31")
        axis.set_xticks(x, [str(seed) for seed in args.seeds])
        axis.set_xlabel("Seed")
        axis.set_ylabel("Last-five mean return")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_root / "phaseVIC_multiseed_returns.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_root / "phaseVIC_multiseed_returns.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
