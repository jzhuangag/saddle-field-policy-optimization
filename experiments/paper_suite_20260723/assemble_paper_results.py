"""Assemble frozen final artifacts for the revised manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RESULTS = HERE / "results"
LINEAR = RESULTS / "linear-geometry-20260723-103816"
TABULAR = RESULTS / "tabular-exact-gap-20260723-113009"
NEURAL = RESULTS / "neural-exact-gap-20260723-113325"
OUTPUT_PDF = PROJECT / "output" / "pdf"
OUTPUT_DATA = PROJECT / "output" / "data"
METHODS = ("QP+G", "noG", "GDA", "Adam-GDA", "EGM", "PPM-3")
STYLES = {"QP+G": ("black", "-"), "noG": ("tab:green", "-"), "GDA": ("tab:orange", "--"), "Adam-GDA": ("tab:blue", "--"), "EGM": ("tab:purple", "-."), "PPM-3": ("tab:red", ":")}


def read_csv(path: Path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {"seed", "step", "current_return", "hard_br_return", "hard_exploitability", "regularized_gap", "field_norm", "hard_br_residual", "hard_br_iterations", "soft_br_residual", "soft_br_iterations", "rotation", "cos_fg", "d", "beta", "gamma", "gamma_active", "g_contribution", "predicted_decrease", "realized_decrease", "inflation", "backtracks", "max_soft_residual", "analytic_d", "grad_norm", "merit", "mu", "ratio", "sigma", "skew_ratio", "z0", "z1"}
    for row in rows:
        for key in numeric & row.keys():
            if row[key] != "":
                row[key] = float(row[key])
    return rows


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(curves, diagnostics):
    summaries = []
    decisions = []
    environments = list(dict.fromkeys(row["environment"] for row in curves))
    for environment in environments:
        seeds = sorted({int(row["seed"]) for row in curves if row["environment"] == environment})
        for method in METHODS:
            final = []
            for seed in seeds:
                selected = sorted([r for r in curves if r["environment"] == environment and r["method"] == method and int(r["seed"]) == seed], key=lambda r: r["step"])
                final.append(selected[-1])
            for metric in ("hard_br_return", "hard_exploitability", "regularized_gap", "field_norm"):
                values = np.array([r[metric] for r in final], dtype=float)
                if metric == "hard_br_return":
                    record = {"environment": environment, "method": method, "seed_count": len(seeds)}
                record[metric + "_mean"] = float(values.mean())
                record[metric + "_sem"] = float(values.std(ddof=1) / math.sqrt(len(values)))
            summaries.append(record)
        qpg = []; nog = []
        for seed in seeds:
            qpg.append(sorted([r for r in curves if r["environment"] == environment and r["method"] == "QP+G" and int(r["seed"]) == seed], key=lambda r: r["step"])[-1])
            nog.append(sorted([r for r in curves if r["environment"] == environment and r["method"] == "noG" and int(r["seed"]) == seed], key=lambda r: r["step"])[-1])
        differences = np.array([a["hard_br_return"] - b["hard_br_return"] for a, b in zip(qpg, nog)])
        sem = float(differences.std(ddof=1) / math.sqrt(len(differences)))
        critical = float(stats.t.ppf(0.975, len(differences) - 1))
        confidence = [float(differences.mean() - critical * sem), float(differences.mean() + critical * sem)]
        wins = int(np.sum(differences > 0)); sign_p = float(stats.binomtest(wins, len(differences), 0.5, alternative="greater").pvalue)
        qpg_exploit = float(np.mean([r["hard_exploitability"] for r in qpg])); nog_exploit = float(np.mean([r["hard_exploitability"] for r in nog]))
        diag = [r for r in diagnostics if r["environment"] == environment]
        decisions.append({"environment": environment, "seed_count": len(seeds), "br_wins": wins, "paired_br_gain_mean": float(differences.mean()), "paired_br_gain_95ci": confidence, "one_sided_exact_sign_test_p": sign_p, "qpg_exploitability_mean": qpg_exploit, "nog_exploitability_mean": nog_exploit, "strict_all_seed_positive": bool(wins == len(seeds) and qpg_exploit < nog_exploit), "confirmatory_positive": bool(confidence[0] > 0 and sign_p < 0.05 and qpg_exploit < nog_exploit), "median_rotation": float(np.median([r["rotation"] for r in diag])), "gamma_activation": float(np.mean([r["gamma_active"] for r in diag])), "mean_g_contribution": float(np.mean([r["g_contribution"] for r in diag])), "inflation_fraction": float(np.mean([r["inflation"] > 1e-10 for r in diag]))})
    return summaries, decisions


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot(curves, environments, destination, figsize):
    panels = (("hard_br_return", "hard BR return"), ("hard_exploitability", "hard exploitability"), ("regularized_gap", "regularized Nash gap"), ("field_norm", "regularized field norm"))
    seeds = sorted({int(row["seed"]) for row in curves})
    fig, axes = plt.subplots(len(environments), 4, figsize=figsize, squeeze=False)
    for row_index, environment in enumerate(environments):
        x = np.array(sorted({r["step"] for r in curves if r["environment"] == environment}))
        for col, (key, title) in enumerate(panels):
            axis = axes[row_index, col]; method_stats = {}
            for method in METHODS:
                data = []
                for seed in seeds:
                    selected = sorted([r for r in curves if r["environment"] == environment and r["method"] == method and int(r["seed"]) == seed], key=lambda r: r["step"])
                    data.append([r[key] for r in selected])
                data = np.asarray(data, dtype=float)
                method_stats[method] = (data.mean(0), data.std(0, ddof=1) / math.sqrt(len(data)))
            stable_low = min(float(np.min(mean - sem)) for method, (mean, sem) in method_stats.items() if method != "Adam-GDA")
            stable_high = max(float(np.max(mean + sem)) for method, (mean, sem) in method_stats.items() if method != "Adam-GDA")
            margin = max(0.08 * (stable_high - stable_low), 1e-6); lower, upper = stable_low - margin, stable_high + margin
            for method in METHODS:
                mean, sem = method_stats[method]
                shown = np.clip(mean, lower, upper) if method == "Adam-GDA" else mean
                shown_sem = np.minimum(sem, np.maximum(upper - shown, 0)) if method == "Adam-GDA" else sem
                axis.plot(x, shown, label=method, color=STYLES[method][0], linestyle=STYLES[method][1], linewidth=1.65)
                axis.fill_between(x, np.maximum(shown - shown_sem, lower), np.minimum(shown + shown_sem, upper), color=STYLES[method][0], alpha=0.08)
                if method == "Adam-GDA":
                    high, low = mean > upper, mean < lower
                    if np.any(high): axis.scatter(x[high], np.full(np.sum(high), upper), marker="^", s=16, color=STYLES[method][0], zorder=4)
                    if np.any(low): axis.scatter(x[low], np.full(np.sum(low), lower), marker="v", s=16, color=STYLES[method][0], zorder=4)
            axis.set(title=f"{environment}: {title}", xlabel="simultaneous joint update", ylim=(lower, upper))
            axis.title.set_fontsize(9.5); axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.016))
    fig.text(0.5, 0.006, "Mean +/- one standard error. Boundary triangles denote clipped Adam-GDA means; CSV values are not clipped.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(destination, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_linear_geometry(curves, destination):
    steps = int(max(row["step"] for row in curves))
    ratio_values = sorted(
        {
            row["ratio"]
            for row in curves
            if math.isfinite(row["ratio"])
        }
    )
    ratio_data = [
        row
        for row in curves
        if math.isfinite(row["ratio"]) and int(row["step"]) == steps
    ]
    rotation_rows = [row for row in curves if not math.isfinite(row["ratio"])]

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.25))
    for method in METHODS:
        selected = sorted(
            [row for row in ratio_data if row["method"] == method],
            key=lambda row: row["ratio"],
        )
        normalized = []
        for row in selected:
            initial = next(
                item["merit"]
                for item in curves
                if item["method"] == method
                and item["ratio"] == row["ratio"]
                and int(item["step"]) == 0
            )
            normalized.append(max(row["merit"] / initial, 1e-30))
        axes[0].semilogy(
            [row["ratio"] for row in selected],
            normalized,
            label=method,
            color=STYLES[method][0],
            linestyle=STYLES[method][1],
        )
    axes[0].axvline(1.0, color="0.5", linewidth=1.0, linestyle=":")
    axes[0].set(
        xlabel=r"rotation ratio $\sigma/\mu$",
        ylabel=r"final $V/V_0$ (log scale)",
        title="phase diagram",
    )

    analytic, numeric = [], []
    for ratio in ratio_values:
        row = next(
            item
            for item in curves
            if item["method"] == "QP+G"
            and item["ratio"] == ratio
            and int(item["step"]) == 0
        )
        scale = (
            (row["mu"] ** 2 + row["sigma"] ** 2) ** 2
            * (row["z0"] ** 2 + row["z1"] ** 2)
        )
        analytic.append(row["analytic_d"] / max(scale, 1e-15))
        numeric.append(row["d"] / max(scale, 1e-15))
    axes[1].plot(ratio_values, analytic, color="black", label="analytic")
    axes[1].plot(
        ratio_values,
        numeric,
        color="tab:red",
        linestyle="--",
        label="computed",
    )
    axes[1].axhline(0.0, color="0.5", linewidth=1.0)
    axes[1].axvline(1.0, color="0.5", linewidth=1.0, linestyle=":")
    axes[1].set(
        xlabel=r"rotation ratio $\sigma/\mu$",
        ylabel="normalized curvature descent $d$",
        title="exact skew threshold",
    )

    for method in METHODS:
        selected = sorted(
            [row for row in rotation_rows if row["method"] == method],
            key=lambda row: row["step"],
        )
        initial = selected[0]["merit"]
        axes[2].semilogy(
            [row["step"] for row in selected],
            [max(row["merit"] / initial, 1e-30) for row in selected],
            label=method,
            color=STYLES[method][0],
            linestyle=STYLES[method][1],
        )
    axes[2].set(
        xlabel="joint update", ylabel=r"$V_k/V_0$", title="pure rotation"
    )

    selected = [
        row
        for row in rotation_rows
        if row["method"] == "QP+G" and int(row["step"]) < steps
    ]
    axes[3].scatter(
        [row["predicted_decrease"] for row in selected],
        [row["realized_decrease"] for row in selected],
        s=16,
        color="black",
    )
    limit = max([row["predicted_decrease"] for row in selected] + [1e-12])
    axes[3].plot(
        [0.0, limit],
        [0.0, limit],
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )
    axes[3].set(
        xlabel="QP predicted decrease",
        ylabel="realized decrease",
        title="quadratic-model identity",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(destination, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_PDF.mkdir(parents=True, exist_ok=True); OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    linear_curves = read_csv(LINEAR / "curves.csv")
    plot_linear_geometry(linear_curves, OUTPUT_PDF / "fig_vi_a_geometry.pdf")
    tab_curves, tab_diag = read_csv(TABULAR / "curves.csv"), read_csv(TABULAR / "diagnostics.csv")
    neu_curves, neu_diag = read_csv(NEURAL / "curves.csv"), read_csv(NEURAL / "diagnostics.csv")
    tab_summary, tab_decisions = summarize(tab_curves, tab_diag)
    neu_summary, neu_decisions = summarize(neu_curves, neu_diag)
    write_csv(OUTPUT_DATA / "vi_b_tabular_summary.csv", tab_summary)
    write_csv(OUTPUT_DATA / "vi_c_neural_summary.csv", neu_summary)
    plot(tab_curves, ["RPS", "CyclicControl", "FrequencyHopping"], OUTPUT_PDF / "fig_vi_b_tabular.pdf", (14.4, 8.9))
    plot(neu_curves, ["CyclicControl", "FrequencyHopping", "RoutingInterdiction"], OUTPUT_PDF / "fig_vi_c_neural.pdf", (14.4, 8.9))
    plot(neu_curves, ["SecurityPatrol", "PursuitEvasion"], OUTPUT_PDF / "fig_vi_c_additional.pdf", (14.4, 6.1))
    source_files = [LINEAR / "curves.csv", LINEAR / "summary.json", TABULAR / "curves.csv", TABULAR / "diagnostics.csv", TABULAR / "summary.json", NEURAL / "curves.csv", NEURAL / "diagnostics.csv", NEURAL / "summary.json"]
    manifest = {"frozen_sources": {str(path.relative_to(PROJECT)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in source_files}, "vi_b_decisions": tab_decisions, "vi_c_decisions": neu_decisions, "main_vi_c_environments": ["CyclicControl", "FrequencyHopping", "RoutingInterdiction"], "additional_confirmed_environment": "SecurityPatrol", "nonconfirmed_environment": "PursuitEvasion"}
    (OUTPUT_DATA / "final_experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2)); print("OUTPUT_PDF=" + str(OUTPUT_PDF)); print("OUTPUT_DATA=" + str(OUTPUT_DATA))


if __name__ == "__main__":
    main()
