"""Merge QP/noG and fixed-baseline finite-trajectory robustness results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ENVIRONMENTS = (
    "CyclicControl",
    "FrequencyHopping",
    "RoutingInterdiction",
    "SecurityPatrol",
)
ETAS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SEEDS = tuple(range(6100, 6105))
METHODS = ("QP+G", "noG", "GDA", "Adam-GDA", "EGM", "PPM-3")
BASELINES = METHODS[1:]
STEPS = 60
CHECKPOINTS = tuple(range(0, STEPS + 1, 10))
TRANSITION_BATCH = 8192
TRAJECTORIES_PER_UPDATE = 512
HORIZON = 16


def source_file(directory: Path, stem: str) -> Path:
    for name in (f"{stem}.csv", f"{stem}.partial.csv"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {stem} data in {directory}")


def read_rows(path: Path) -> list[dict]:
    integer_fields = {
        "seed",
        "step",
        "transition_batch",
        "trajectories_per_update",
        "horizon",
        "cumulative_transitions",
        "backtracks",
        "transitions_used",
        "hard_br_iterations",
        "soft_br_iterations",
        "game_value_iterations",
        "field_queries",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    converted: list[dict] = []
    for original in source:
        row = dict(original)
        for key, value in row.items():
            if key in {"environment", "method"}:
                continue
            row[key] = int(float(value)) if key in integer_fields else float(value)
        converted.append(row)
    return converted


def key(row: dict) -> tuple:
    return (
        row["environment"],
        float(row["eta"]),
        int(row["seed"]),
        row["method"],
        int(row["step"]),
    )


def merge_unique(groups: list[list[dict]], label: str) -> list[dict]:
    indexed: dict[tuple, dict] = {}
    for rows in groups:
        for row in rows:
            row_key = key(row)
            if row_key in indexed and indexed[row_key] != row:
                raise RuntimeError(f"conflicting duplicate {label}: {row_key}")
            indexed[row_key] = row
    return [indexed[row_key] for row_key in sorted(indexed)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sem_ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / math.sqrt(len(array)))
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * sem)
    return mean, sem, mean - half_width, mean + half_width


def validate(rows: list[dict], diagnostics: list[dict]) -> dict:
    expected_curves = {
        (environment, eta, seed, method, step)
        for environment in ENVIRONMENTS
        for eta in ETAS
        for seed in SEEDS
        for method in METHODS
        for step in CHECKPOINTS
    }
    expected_diagnostics = {
        (environment, eta, seed, method, step)
        for environment in ENVIRONMENTS
        for eta in ETAS
        for seed in SEEDS
        for method in METHODS
        for step in range(1, STEPS + 1)
    }
    actual_curves = {key(row) for row in rows}
    actual_diagnostics = {key(row) for row in diagnostics}
    if actual_curves != expected_curves or len(rows) != len(expected_curves):
        raise RuntimeError(
            f"curve grid mismatch: missing={sorted(expected_curves-actual_curves)[:5]}, "
            f"extra={sorted(actual_curves-expected_curves)[:5]}"
        )
    if (
        actual_diagnostics != expected_diagnostics
        or len(diagnostics) != len(expected_diagnostics)
    ):
        raise RuntimeError(
            "diagnostic grid mismatch: "
            f"missing={sorted(expected_diagnostics-actual_diagnostics)[:5]}, "
            f"extra={sorted(actual_diagnostics-expected_diagnostics)[:5]}"
        )
    for label, data in (("curve", rows), ("diagnostic", diagnostics)):
        for row in data:
            for field, value in row.items():
                if field in {"environment", "method"}:
                    continue
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"nonfinite {label} value {field} at {key(row)}")
    for row in rows:
        if int(row["transition_batch"]) != TRANSITION_BATCH:
            raise RuntimeError(f"transition batch mismatch at {key(row)}")
        if int(row["trajectories_per_update"]) != TRAJECTORIES_PER_UPDATE:
            raise RuntimeError(f"trajectory count mismatch at {key(row)}")
        if int(row["horizon"]) != HORIZON:
            raise RuntimeError(f"horizon mismatch at {key(row)}")
        if int(row["cumulative_transitions"]) != int(row["step"]) * TRANSITION_BATCH:
            raise RuntimeError(f"cumulative transition mismatch at {key(row)}")

    initial = {
        (row["environment"], float(row["eta"]), int(row["seed"]), row["method"]): row
        for row in rows
        if int(row["step"]) == 0
    }
    metric_names = (
        "current_return",
        "hard_br_return",
        "hard_exploitability",
        "regularized_gap",
        "field_norm",
    )
    maximum_initial_difference = 0.0
    for environment in ENVIRONMENTS:
        for eta in ETAS:
            for seed in SEEDS:
                reference = initial[(environment, eta, seed, "QP+G")]
                for method in METHODS[1:]:
                    candidate = initial[(environment, eta, seed, method)]
                    maximum_initial_difference = max(
                        maximum_initial_difference,
                        max(
                            abs(float(reference[name]) - float(candidate[name]))
                            for name in metric_names
                        ),
                    )
    if maximum_initial_difference > 1.0e-12:
        raise RuntimeError(
            f"methods do not share initialization: {maximum_initial_difference}"
        )
    return {
        "curve_rows": len(rows),
        "diagnostic_rows": len(diagnostics),
        "maximum_initial_metric_difference": maximum_initial_difference,
        "maximum_hard_br_residual": max(float(row["hard_br_residual"]) for row in rows),
        "maximum_soft_br_residual": max(float(row["soft_br_residual"]) for row in rows),
        "maximum_game_value_residual": max(float(row["game_value_residual"]) for row in rows),
    }


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    comparisons: list[dict] = []
    metric_names = ("hard_br_return", "hard_exploitability")
    final = [row for row in rows if int(row["step"]) == STEPS]
    for environment in ENVIRONMENTS:
        for eta in ETAS:
            by_method: dict[str, list[dict]] = {}
            for method in METHODS:
                selected = sorted(
                    (
                        row
                        for row in final
                        if row["environment"] == environment
                        and math.isclose(float(row["eta"]), eta)
                        and row["method"] == method
                    ),
                    key=lambda row: int(row["seed"]),
                )
                if [int(row["seed"]) for row in selected] != list(SEEDS):
                    raise RuntimeError(f"incomplete final rows for {environment}, {eta}, {method}")
                by_method[method] = selected
                summary = {
                    "environment": environment,
                    "eta": eta,
                    "method": method,
                    "seed_count": len(selected),
                }
                for metric in metric_names:
                    mean, sem, lower, upper = mean_sem_ci(
                        [float(row[metric]) for row in selected]
                    )
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_sem"] = sem
                    summary[f"{metric}_ci95_lower"] = lower
                    summary[f"{metric}_ci95_upper"] = upper
                summaries.append(summary)

            proposed = by_method["QP+G"]
            for baseline in BASELINES:
                comparator = by_method[baseline]
                return_gain = [
                    float(left["hard_br_return"]) - float(right["hard_br_return"])
                    for left, right in zip(proposed, comparator)
                ]
                exploitability_reduction = [
                    float(right["hard_exploitability"])
                    - float(left["hard_exploitability"])
                    for left, right in zip(proposed, comparator)
                ]
                comparison = {
                    "environment": environment,
                    "eta": eta,
                    "baseline": baseline,
                    "seed_count": len(SEEDS),
                }
                for name, values in (
                    ("return_gain", return_gain),
                    ("exploitability_reduction", exploitability_reduction),
                ):
                    mean, sem, lower, upper = mean_sem_ci(values)
                    comparison[f"{name}_mean"] = mean
                    comparison[f"{name}_sem"] = sem
                    comparison[f"{name}_ci95_lower"] = lower
                    comparison[f"{name}_ci95_upper"] = upper
                    comparison[f"{name}_positive_seed_count"] = sum(
                        value > 0.0 for value in values
                    )
                comparisons.append(comparison)
    return summaries, comparisons


def plot(summaries: list[dict], output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.weight": "normal",
            "font.size": 7.2,
            "mathtext.fontset": "stix",
            "axes.titlesize": 7.2,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.2,
            "axes.labelweight": "normal",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    titles = {environment: environment for environment in ENVIRONMENTS}
    styles = {
        "QP+G": ("black", "-"),
        "noG": ("tab:green", "-"),
        "GDA": ("tab:orange", "--"),
        "Adam-GDA": ("tab:blue", "--"),
        "EGM": ("tab:purple", "-."),
        "PPM-3": ("tab:red", ":"),
    }
    figure, axes = plt.subplots(2, 4, figsize=(7.16, 2.75), sharex="col")
    for column, environment in enumerate(ENVIRONMENTS):
        for row_index, (metric, ylabel) in enumerate(
            (("hard_br_return", "Worst-case return"), ("hard_exploitability", "Exploitability"))
        ):
            axis = axes[row_index, column]
            for method in METHODS:
                selected = sorted(
                    (
                        row
                        for row in summaries
                        if row["environment"] == environment and row["method"] == method
                    ),
                    key=lambda row: float(row["eta"]),
                )
                x = np.asarray([float(row["eta"]) for row in selected])
                mean = np.asarray([float(row[f"{metric}_mean"]) for row in selected])
                sem = np.asarray([float(row[f"{metric}_sem"]) for row in selected])
                color, linestyle = styles[method]
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.35,
                    label=method,
                    zorder=4 if method == "QP+G" else 2,
                )
                axis.fill_between(
                    x,
                    mean - sem,
                    mean + sem,
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
            axis.axvline(1.0, color="0.45", linestyle=":", linewidth=0.85)
            axis.grid(True, color="0.84", linewidth=0.45, alpha=0.55)
            axis.set_axisbelow(True)
            axis.tick_params(width=0.65, length=2.6, pad=1.5)
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
            if row_index == 0:
                axis.set_title(titles[environment])
            if column == 0:
                axis.set_ylabel(ylabel, labelpad=2.0)
            if row_index == 1:
                axis.set_xlabel(r"Attack strength $\eta$", labelpad=1.5)
            axis.set_xticks(ETAS[::2])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
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
    figure.subplots_adjust(
        left=0.080,
        right=0.995,
        top=0.810,
        bottom=0.145,
        wspace=0.56,
        hspace=0.34,
    )
    figure.savefig(output_pdf, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(output_png, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-source", type=Path, required=True)
    parser.add_argument("baseline_sources", type=Path, nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-output-root", type=Path, required=True)
    args = parser.parse_args()

    curve_groups = [read_rows(source_file(args.proposed_source, "curves"))]
    diagnostic_groups = [read_rows(source_file(args.proposed_source, "diagnostics"))]
    protocols: list[dict] = []
    for source in args.baseline_sources:
        curve_groups.append(read_rows(source_file(source, "curves")))
        diagnostic_groups.append(read_rows(source_file(source, "diagnostics")))
        protocol_path = source / "protocol.json"
        if not protocol_path.exists():
            raise FileNotFoundError(f"missing baseline protocol: {protocol_path}")
        protocols.append(json.loads(protocol_path.read_text(encoding="utf-8")))
    for protocol in protocols:
        if not math.isclose(float(protocol["shared_learning_rate"]), 0.03):
            raise RuntimeError("unexpected shared baseline learning rate")
        if not math.isclose(float(protocol["adam_learning_rate"]), 0.001):
            raise RuntimeError("unexpected Adam-GDA learning rate")
        if not protocol["equal_transition_budget_per_update"]:
            raise RuntimeError("baseline run did not enforce equal transition budget")

    rows = merge_unique(curve_groups, "curve")
    diagnostics = merge_unique(diagnostic_groups, "diagnostic")
    validation = validate(rows, diagnostics)
    summaries, comparisons = summarize(rows)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output_root / f"finite-trajectory-six-methods-{timestamp}"
    output.mkdir(parents=True)
    write_csv(output / "curves.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "qp_comparisons.csv", comparisons)
    figure_pdf = output / "finite_trajectory_six_methods_2x4.pdf"
    figure_png = output / "finite_trajectory_six_methods_2x4.png"
    plot(summaries, figure_pdf, figure_png)
    args.paper_output_root.mkdir(parents=True, exist_ok=True)
    paper_pdf = args.paper_output_root / "fig_vi_d_finite_trajectory_robustness_six_methods.pdf"
    paper_png = args.paper_output_root / "fig_vi_d_finite_trajectory_robustness_six_methods.png"
    shutil.copy2(figure_pdf, paper_pdf)
    shutil.copy2(figure_png, paper_png)

    positive = {
        "return": sum(float(row["return_gain_mean"]) > 0.0 for row in comparisons),
        "exploitability": sum(
            float(row["exploitability_reduction_mean"]) > 0.0
            for row in comparisons
        ),
        "total_comparisons": len(comparisons),
    }
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run+validate",
            "origin_date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "verification_status": "ANALYZED",
            "version_label": "exp_result_v1",
        },
        "protocol": {
            "environments": ENVIRONMENTS,
            "attack_strength_values": ETAS,
            "seeds": SEEDS,
            "methods": METHODS,
            "steps": STEPS,
            "trajectories_per_update": TRAJECTORIES_PER_UPDATE,
            "horizon": HORIZON,
            "shared_baseline_learning_rate": 0.03,
            "adam_learning_rate": 0.001,
        },
        "validation": validation,
        "qp_positive_mean_comparisons": positive,
        "outputs": {
            "result_directory": str(output),
            "paper_pdf": str(paper_pdf),
            "paper_png": str(paper_png),
        },
    }
    (output / "experiment_result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"OUTPUT={output}")
    print(json.dumps({"validation": validation, "positive": positive}, indent=2))


if __name__ == "__main__":
    main()
