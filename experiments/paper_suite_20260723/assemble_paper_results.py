"""Regenerate journal figures and summaries from frozen release artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIG_FONT_SIZE = 7.2
METHOD_LINE_WIDTH = 1.35
REFERENCE_LINE_WIDTH = 0.85
BOUNDARY_MARKER_SIZE = 15
CI_ALPHA = 0.08
SPINE_WIDTH = 0.65
GRID_LINE_WIDTH = 0.45
GRID_ALPHA = 0.55
DOUBLE_COLUMN_WIDTH = 7.16
GAMMA_MAX = 0.03

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.weight": "normal",
        "font.size": FIG_FONT_SIZE,
        "mathtext.fontset": "stix",
        "axes.titlesize": FIG_FONT_SIZE,
        "axes.titleweight": "normal",
        "axes.labelsize": FIG_FONT_SIZE,
        "axes.labelweight": "normal",
        "xtick.labelsize": FIG_FONT_SIZE,
        "ytick.labelsize": FIG_FONT_SIZE,
        "legend.fontsize": FIG_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RESULTS = HERE / "results"
LINEAR = RESULTS / "linear-geometry-20260823-234632"
TABULAR = RESULTS / "tabular-exact-gap-20260723-113009"
NEURAL = RESULTS / "neural-journal-four-20260824"
OUTPUT_PDF = PROJECT / "output" / "pdf"
OUTPUT_DATA = PROJECT / "output" / "data"
METHODS = ("QP+G", "noG", "GDA", "Adam-GDA", "EGM", "PPM-3")
STYLES = {"QP+G": ("black", "-"), "noG": ("tab:green", "-"), "GDA": ("tab:orange", "--"), "Adam-GDA": ("tab:blue", "--"), "EGM": ("tab:purple", "-."), "PPM-3": ("tab:red", ":")}
MARKERS = {"QP+G": "o", "noG": "s", "GDA": "^", "Adam-GDA": "v", "EGM": "D", "PPM-3": "P"}


def style_axis(axis):
    axis.grid(
        True,
        color="0.84",
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_ALPHA,
    )
    axis.set_axisbelow(True)
    axis.tick_params(width=SPINE_WIDTH, length=2.6, pad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def read_csv(path: Path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {"seed", "step", "current_return", "hard_br_return", "hard_exploitability", "regularized_gap", "field_norm", "hard_br_residual", "hard_br_iterations", "soft_br_residual", "soft_br_iterations", "rotation", "cos_fg", "d", "beta", "gamma", "gamma_active", "g_contribution", "curvature_contribution", "curvature_norm", "predicted_decrease", "realized_decrease", "inflation", "backtracks", "max_soft_residual", "analytic_d", "grad_norm", "merit", "mu", "ratio", "sigma", "skew_ratio", "z0", "z1"}
    for row in rows:
        for key in numeric & row.keys():
            if row[key] != "":
                row[key] = float(row[key])
    return rows


def canonical_bytes(path: Path) -> bytes:
    """Return text bytes with platform-dependent line endings normalized."""
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json"}:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


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
    with path.open("w", newline="\n", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(curves, environments, destination):
    panels = (
        ("hard_br_return", "Worst-case return"),
        ("hard_exploitability", "Exploitability"),
        ("regularized_gap", "Regularized Nash gap"),
        ("field_norm", r"Field norm $\|\mathbf{F}\|$"),
    )
    seeds = sorted({int(row["seed"]) for row in curves})
    # Preserve the configured font sizes and provide enough vertical room for
    # row labels, tick labels, and axis labels to remain visually separated.
    figure_height = 1.55 * len(environments)
    fig, axes = plt.subplots(
        len(environments),
        4,
        figsize=(DOUBLE_COLUMN_WIDTH, figure_height),
        squeeze=False,
    )
    for row_index, environment in enumerate(environments):
        x = np.array(sorted({r["step"] for r in curves if r["environment"] == environment}))
        for col, (key, ylabel) in enumerate(panels):
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
                axis.plot(
                    x,
                    shown,
                    label=method,
                    color=STYLES[method][0],
                    linestyle=STYLES[method][1],
                    linewidth=METHOD_LINE_WIDTH,
                )
                axis.fill_between(
                    x,
                    np.maximum(shown - shown_sem, lower),
                    np.minimum(shown + shown_sem, upper),
                    color=STYLES[method][0],
                    alpha=CI_ALPHA,
                    linewidth=0,
                )
                if method == "Adam-GDA":
                    high, low = mean > upper, mean < lower
                    if np.any(high):
                        axis.scatter(
                            x[high],
                            np.full(np.sum(high), upper),
                            marker="^",
                            s=BOUNDARY_MARKER_SIZE,
                            color=STYLES[method][0],
                            linewidths=0,
                            zorder=4,
                        )
                    if np.any(low):
                        axis.scatter(
                            x[low],
                            np.full(np.sum(low), lower),
                            marker="v",
                            s=BOUNDARY_MARKER_SIZE,
                            color=STYLES[method][0],
                            linewidths=0,
                            zorder=4,
                        )
            axis.set_ylim(lower, upper)
            axis.set_xlabel(r"Joint update $k$", labelpad=1.5)
            axis.set_ylabel(ylabel, labelpad=2.0)
            style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.25,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0,
    )
    fig.subplots_adjust(
        left=0.080,
        right=0.995,
        top=0.925,
        bottom=0.105,
        wspace=0.56,
        hspace=0.80,
    )
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for row_index, environment in enumerate(environments):
        left = axes[row_index, 0].get_position()
        right = axes[row_index, -1].get_position()
        if row_index + 1 < len(environments):
            next_row = axes[row_index + 1, 0].get_position()
            label_y = 0.5 * (left.y0 + next_row.y1)
        else:
            label_y = left.y0 - 0.070
        fig.text(
            0.5 * (left.x0 + right.x1),
            label_y,
            f"({panel_letters[row_index]}) {environment}",
            ha="center",
            va="top",
            fontweight="normal",
        )
    fig.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_population_summary(tabular_summary, neural_summary, destination):
    """Plot the decisive population-oracle metric in a compact two-panel figure."""
    panels = (
        (
            tabular_summary,
            ("RPS", "CyclicControl", "FrequencyHopping"),
            ("RPS", "Cyclic", "Frequency"),
            "(a) Tabular policies",
        ),
        (
            neural_summary,
            ("CyclicControl", "FrequencyHopping", "RoutingInterdiction"),
            ("Cyclic", "Frequency", "Routing"),
            "(b) Neural policies",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 1.95))
    offsets = np.linspace(-0.25, 0.25, len(METHODS))
    for axis, (summary, environments, tick_labels, panel_label) in zip(axes, panels):
        base = np.arange(len(environments), dtype=float)
        for offset, method in zip(offsets, METHODS):
            selected = [
                next(
                    row
                    for row in summary
                    if row["environment"] == environment and row["method"] == method
                )
                for environment in environments
            ]
            mean = np.asarray(
                [row["hard_exploitability_mean"] for row in selected], dtype=float
            )
            sem = np.asarray(
                [row["hard_exploitability_sem"] for row in selected], dtype=float
            )
            lower = np.minimum(sem, 0.8 * mean)
            axis.errorbar(
                base + offset,
                mean,
                yerr=np.vstack((lower, sem)),
                color=STYLES[method][0],
                marker=MARKERS[method],
                markersize=3.2,
                linestyle="none",
                linewidth=METHOD_LINE_WIDTH,
                elinewidth=REFERENCE_LINE_WIDTH,
                capsize=1.8,
                capthick=REFERENCE_LINE_WIDTH,
                label=method,
                zorder=4 if method == "QP+G" else 2,
            )
        axis.set_yscale("log")
        axis.set_xticks(base)
        axis.set_xticklabels(tick_labels)
        axis.set_xlim(-0.48, len(environments) - 0.52)
        axis.set_ylabel("Final exploitability", labelpad=2.0)
        axis.grid(
            True,
            axis="y",
            color="0.84",
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_ALPHA,
        )
        axis.set_axisbelow(True)
        axis.tick_params(width=SPINE_WIDTH, length=2.6, pad=1.5)
        for spine in axis.spines.values():
            spine.set_linewidth(SPINE_WIDTH)
        position = axis.get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            0.055,
            panel_label,
            ha="center",
            va="bottom",
            fontweight="normal",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.4,
        columnspacing=0.85,
        handletextpad=0.35,
        borderaxespad=0,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.78,
        bottom=0.28,
        wspace=0.25,
    )
    fig.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination.with_suffix(".png"), dpi=300)
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

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(DOUBLE_COLUMN_WIDTH, 1.95),
    )
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
            linewidth=METHOD_LINE_WIDTH,
        )
    axes[0].axvline(
        1.0,
        color="0.45",
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle=":",
    )
    axes[0].set(
        xlabel=r"Rotation ratio $\sigma/\mu$",
        ylabel=r"Final field energy $\phi_K/\phi_0$",
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
    analytic_line = axes[1].plot(
        ratio_values,
        analytic,
        color="0.20",
        linewidth=METHOD_LINE_WIDTH,
        label="Analytical $d$",
    )[0]
    numerical_line = axes[1].plot(
        ratio_values,
        numeric,
        color="tab:red",
        linestyle="--",
        linewidth=METHOD_LINE_WIDTH,
        label="Numerical $d$",
    )[0]
    axes[1].axhline(0.0, color="0.45", linewidth=REFERENCE_LINE_WIDTH)
    axes[1].axvline(
        1.0,
        color="0.45",
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle=":",
    )
    axes[1].set(
        xlabel=r"Rotation ratio $\sigma/\mu$",
        ylabel="Normalized curvature descent $d$",
    )
    axes[1].legend(
        handles=[analytic_line, numerical_line],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
        handlelength=1.35,
        handletextpad=0.35,
        borderpad=0.20,
        borderaxespad=0.25,
        labelspacing=0.18,
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
            linewidth=METHOD_LINE_WIDTH,
        )
    axes[2].set(
        xlabel=r"Joint update $k$",
        ylabel=r"Normalized field energy $\phi_k/\phi_0$",
    )

    panel_d_rows = sorted(
        [
            row
            for row in curves
            if row["method"] == "QP+G"
            and math.isfinite(row["ratio"])
            and int(row["step"]) == 0
        ],
        key=lambda row: row["ratio"],
    )
    panel_d_ratios = np.asarray(
        [row["ratio"] for row in panel_d_rows], dtype=float
    )
    normalized_gamma = np.asarray(
        [row["gamma"] / GAMMA_MAX for row in panel_d_rows], dtype=float
    )
    curvature_contribution = np.asarray(
        [row["curvature_contribution"] for row in panel_d_rows], dtype=float
    )
    axes[3].plot(
        panel_d_ratios,
        normalized_gamma,
        color="black",
        linewidth=METHOD_LINE_WIDTH,
        marker="o",
        markersize=2.8,
        markevery=5,
        label=r"$\gamma_0/\gamma_{\max}$",
    )
    axes[3].plot(
        panel_d_ratios,
        curvature_contribution,
        color="tab:purple",
        linestyle="--",
        linewidth=METHOD_LINE_WIDTH,
        marker="s",
        markersize=2.6,
        markevery=5,
        label=r"$C_0^G$",
    )
    axes[3].axvline(
        1.0,
        color="0.45",
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle=":",
    )
    axes[3].set(
        xlabel=r"Rotation ratio $\sigma/\mu$",
        ylabel="Normalized curvature use",
        ylim=(0.0, 1.0),
    )
    axes[3].set_yticks([0.0, 0.5, 1.0])
    axes[3].legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.80),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
        handlelength=1.35,
        handletextpad=0.35,
        borderpad=0.20,
        borderaxespad=0.0,
        labelspacing=0.18,
    )
    for axis in axes:
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.8,
        columnspacing=0.75,
        handletextpad=0.35,
        borderaxespad=0,
    )
    fig.subplots_adjust(
        left=0.060,
        right=0.995,
        top=0.77,
        bottom=0.32,
        wspace=0.38,
    )
    panel_labels = (
        "(a) Rotation-ratio sweep",
        "(b) Curvature-descent threshold",
        "(c) Pure-rotation dynamics",
        "(d) Adaptive curvature allocation",
    )
    for axis, panel_label in zip(axes, panel_labels):
        position = axis.get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            position.y0 - 0.205,
            panel_label,
            ha="center",
            va="top",
            fontweight="normal",
        )
    fig.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    plt.close(fig)


def project_relative(path: Path) -> str:
    """Return a portable repository-relative artifact path."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT.resolve())
    except ValueError as error:
        raise ValueError(
            "experiment inputs must be located inside the repository: {}".format(
                resolved
            )
        ) from error
    return str(relative).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate journal figures from selected result directories."
    )
    parser.add_argument("--linear-dir", type=Path, default=LINEAR)
    parser.add_argument("--tabular-dir", type=Path, default=TABULAR)
    parser.add_argument("--neural-dir", type=Path, default=NEURAL)
    parser.add_argument("--output-pdf-dir", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--output-data-dir", type=Path, default=OUTPUT_DATA)
    args = parser.parse_args()

    linear_dir = args.linear_dir.resolve()
    tabular_dir = args.tabular_dir.resolve()
    neural_dir = args.neural_dir.resolve()
    output_pdf = args.output_pdf_dir.resolve()
    output_data = args.output_data_dir.resolve()
    output_pdf.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)

    linear_curves = read_csv(linear_dir / "curves.csv")
    plot_linear_geometry(linear_curves, output_pdf / "fig_vi_a_geometry.pdf")
    tab_curves, tab_diag = read_csv(tabular_dir / "curves.csv"), read_csv(tabular_dir / "diagnostics.csv")
    neu_curves, neu_diag = read_csv(neural_dir / "curves.csv"), read_csv(neural_dir / "diagnostics.csv")
    tab_summary, tab_decisions = summarize(tab_curves, tab_diag)
    neu_summary, neu_decisions = summarize(neu_curves, neu_diag)
    write_csv(output_data / "vi_b_tabular_summary.csv", tab_summary)
    write_csv(output_data / "vi_c_neural_summary.csv", neu_summary)
    plot(tab_curves, ["RPS", "CyclicControl", "FrequencyHopping"], output_pdf / "fig_vi_b_tabular.pdf")
    plot(
        neu_curves,
        [
            "CyclicControl",
            "FrequencyHopping",
            "RoutingInterdiction",
            "SecurityPatrol",
        ],
        output_pdf / "fig_vi_c_neural.pdf",
    )
    plot_population_summary(
        tab_summary,
        neu_summary,
        output_pdf / "fig_vi_bc_population_summary.pdf",
    )
    source_files = [linear_dir / "curves.csv", linear_dir / "summary.json", tabular_dir / "curves.csv", tabular_dir / "diagnostics.csv", tabular_dir / "summary.json", neural_dir / "curves.csv", neural_dir / "diagnostics.csv", neural_dir / "summary.json"]
    manifest = {
        "schema": "journal-artifact-manifest/1",
        "selected_results": {
            "linear": project_relative(linear_dir),
            "tabular": project_relative(tabular_dir),
            "neural": project_relative(neural_dir),
        },
        "frozen_sources": {
            project_relative(path): {
                "sha256": sha256(path),
                "canonical_bytes": len(canonical_bytes(path)),
            }
            for path in source_files
        },
        "vi_b_decisions": tab_decisions,
        "vi_c_decisions": neu_decisions,
        "displayed_vi_c_environments": [
            "CyclicControl",
            "FrequencyHopping",
            "RoutingInterdiction",
            "SecurityPatrol",
        ],
    }
    with (output_data / "final_experiment_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2)); print("OUTPUT_PDF=" + str(output_pdf)); print("OUTPUT_DATA=" + str(output_data))


if __name__ == "__main__":
    main()
