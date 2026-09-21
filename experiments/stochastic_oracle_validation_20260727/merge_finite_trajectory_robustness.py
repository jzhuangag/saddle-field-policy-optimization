"""Merge four-environment finite-trajectory robustness shards.

The input directories may contain completed ``curves.csv`` / ``diagnostics.csv``
files or interruption-safe ``*.partial.csv`` files.  Rows are joined by their
full experimental key, validated against the pre-specified protocol, summarized
over five paired seeds, and rendered as the paper's 2-by-4 robustness figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from finite_trajectory_attack_sweep import METHODS, summarize, write_csv


ENVIRONMENTS = (
    "CyclicControl",
    "FrequencyHopping",
    "RoutingInterdiction",
    "SecurityPatrol",
)
ETAS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SEEDS = tuple(range(6100, 6105))
STEPS = 60
CHECKPOINTS = tuple(range(0, STEPS + 1, 10))
TRANSITION_BATCH = 8192
TRAJECTORIES_PER_UPDATE = 512
HORIZON = 16


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_file(directory: Path, stem: str) -> Path:
    final_path = directory / f"{stem}.csv"
    partial_path = directory / f"{stem}.partial.csv"
    if final_path.exists():
        return final_path
    if partial_path.exists():
        return partial_path
    raise FileNotFoundError(f"neither {final_path} nor {partial_path} exists")


def convert_rows(rows: list[dict]) -> list[dict]:
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
    }
    converted: list[dict] = []
    for source in rows:
        row = dict(source)
        for key, value in row.items():
            if key in {"environment", "method"}:
                continue
            row[key] = int(float(value)) if key in integer_fields else float(value)
        converted.append(row)
    return converted


def row_key(row: dict) -> tuple:
    return (
        row["environment"],
        float(row["eta"]),
        int(row["seed"]),
        row["method"],
        int(row["step"]),
    )


def merge_unique(groups: list[list[dict]], label: str) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for rows in groups:
        for row in rows:
            key = row_key(row)
            if key in merged and merged[key] != row:
                raise RuntimeError(f"conflicting duplicate {label} row: {key}")
            merged[key] = row
    return [merged[key] for key in sorted(merged)]


def expected_keys(steps: tuple[int, ...]) -> set[tuple]:
    return {
        (environment, eta, seed, method, step)
        for environment in ENVIRONMENTS
        for eta in ETAS
        for seed in SEEDS
        for method in METHODS
        for step in steps
    }


def assert_finite(rows: list[dict], label: str) -> None:
    for row in rows:
        for key, value in row.items():
            if key in {"environment", "method"}:
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f"nonfinite {label} value at {row_key(row)}: {key}")


def validate_grid(rows: list[dict], diagnostics: list[dict]) -> dict:
    curve_expected = expected_keys(CHECKPOINTS)
    diagnostic_expected = expected_keys(tuple(range(1, STEPS + 1)))
    curve_actual = {row_key(row) for row in rows}
    diagnostic_actual = {row_key(row) for row in diagnostics}
    if curve_actual != curve_expected or len(rows) != len(curve_expected):
        raise RuntimeError(
            "curve grid mismatch; "
            f"missing={sorted(curve_expected - curve_actual)[:5]}, "
            f"extra={sorted(curve_actual - curve_expected)[:5]}"
        )
    if (
        diagnostic_actual != diagnostic_expected
        or len(diagnostics) != len(diagnostic_expected)
    ):
        raise RuntimeError(
            "diagnostic grid mismatch; "
            f"missing={sorted(diagnostic_expected - diagnostic_actual)[:5]}, "
            f"extra={sorted(diagnostic_actual - diagnostic_expected)[:5]}"
        )
    assert_finite(rows, "curve")
    assert_finite(diagnostics, "diagnostic")
    for row in rows:
        if int(row["transition_batch"]) != TRANSITION_BATCH:
            raise RuntimeError(f"transition-batch mismatch at {row_key(row)}")
        if int(row["trajectories_per_update"]) != TRAJECTORIES_PER_UPDATE:
            raise RuntimeError(f"trajectory-count mismatch at {row_key(row)}")
        if int(row["horizon"]) != HORIZON:
            raise RuntimeError(f"horizon mismatch at {row_key(row)}")
        expected_transitions = int(row["step"]) * TRANSITION_BATCH
        if int(row["cumulative_transitions"]) != expected_transitions:
            raise RuntimeError(f"transition accounting mismatch at {row_key(row)}")

    initial_metrics = (
        "current_return",
        "hard_br_return",
        "hard_exploitability",
        "regularized_gap",
        "field_norm",
    )
    initial = {
        (row["environment"], float(row["eta"]), int(row["seed"]), row["method"]): row
        for row in rows
        if int(row["step"]) == 0
    }
    max_initial_difference = 0.0
    for environment in ENVIRONMENTS:
        for eta in ETAS:
            for seed in SEEDS:
                left = initial[(environment, eta, seed, "QP+G")]
                right = initial[(environment, eta, seed, "noG")]
                max_initial_difference = max(
                    max_initial_difference,
                    max(abs(float(left[name]) - float(right[name])) for name in initial_metrics),
                )
    if max_initial_difference > 1.0e-12:
        raise RuntimeError(
            f"paired methods do not share initialization: {max_initial_difference}"
        )
    return {
        "curve_rows": len(rows),
        "diagnostic_rows": len(diagnostics),
        "max_initial_metric_difference": max_initial_difference,
        "max_hard_br_residual": max(float(row["hard_br_residual"]) for row in rows),
        "max_soft_br_residual": max(float(row["soft_br_residual"]) for row in rows),
        "max_game_value_residual": max(float(row["game_value_residual"]) for row in rows),
    }


def plot_summary(summaries: list[dict], output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    title = {
        "CyclicControl": "Cyclic Control",
        "FrequencyHopping": "Frequency Hopping",
        "RoutingInterdiction": "Routing Interdiction",
        "SecurityPatrol": "Security Patrol",
    }
    styles = {
        "QP+G": {"color": "black", "marker": "o"},
        "noG": {"color": "#2ca25f", "marker": "s"},
    }
    figure, axes = plt.subplots(2, 4, figsize=(7.16, 4.25), sharex="col")
    metrics = (
        ("hard_br_return", "Worst-case return"),
        ("hard_exploitability", "Exploitability"),
    )
    for column, environment in enumerate(ENVIRONMENTS):
        for row_index, (metric, ylabel) in enumerate(metrics):
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
                lower = np.asarray([float(row[f"{metric}_ci95_lower"]) for row in selected])
                upper = np.asarray([float(row[f"{metric}_ci95_upper"]) for row in selected])
                axis.plot(
                    x,
                    mean,
                    color=styles[method]["color"],
                    marker=styles[method]["marker"],
                    linewidth=1.65,
                    markersize=3.8,
                    label=method,
                )
                axis.fill_between(
                    x,
                    lower,
                    upper,
                    color=styles[method]["color"],
                    alpha=0.12,
                    linewidth=0,
                )
            axis.axvline(1.0, color="0.55", linestyle="--", linewidth=0.8)
            axis.grid(alpha=0.22, linewidth=0.6)
            axis.set_title(title[environment] if row_index == 0 else "")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 1:
                axis.set_xlabel(r"Attack strength $\eta$")
            axis.set_xticks(ETAS[::2])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955), w_pad=0.7, h_pad=0.75)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--paper-output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "output" / "pdf",
    )
    args = parser.parse_args()

    curve_groups: list[list[dict]] = []
    diagnostic_groups: list[list[dict]] = []
    source_manifest: list[dict] = []
    for source in args.sources:
        curve_path = source_file(source, "curves")
        diagnostic_path = source_file(source, "diagnostics")
        curve_groups.append(convert_rows(read_csv(curve_path)))
        diagnostic_groups.append(convert_rows(read_csv(diagnostic_path)))
        source_manifest.append(
            {
                "directory": str(source),
                "curves": str(curve_path),
                "curves_sha256": sha256(curve_path),
                "diagnostics": str(diagnostic_path),
                "diagnostics_sha256": sha256(diagnostic_path),
            }
        )
    rows = merge_unique(curve_groups, "curve")
    diagnostics = merge_unique(diagnostic_groups, "diagnostic")
    validation = validate_grid(rows, diagnostics)
    summaries, paired = summarize(
        rows, diagnostics, ENVIRONMENTS, ETAS, SEEDS, STEPS
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output_root / f"finite-trajectory-robustness-4env-{timestamp}"
    output.mkdir(parents=True)
    write_csv(output / "curves.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_summary.csv", paired)
    figure_pdf = output / "finite_trajectory_robustness_2x4.pdf"
    figure_png = output / "finite_trajectory_robustness_2x4.png"
    plot_summary(summaries, figure_pdf, figure_png)

    args.paper_output_root.mkdir(parents=True, exist_ok=True)
    paper_pdf = args.paper_output_root / "fig_vi_d_finite_trajectory_robustness_candidate.pdf"
    paper_png = args.paper_output_root / "fig_vi_d_finite_trajectory_robustness_candidate.png"
    shutil.copy2(figure_pdf, paper_pdf)
    shutil.copy2(figure_png, paper_png)

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
            "methods": METHODS,
            "seeds": SEEDS,
            "steps": STEPS,
            "transitions_per_update": TRANSITION_BATCH,
            "trajectories_per_update": TRAJECTORIES_PER_UPDATE,
            "horizon": HORIZON,
            "training_oracle": "finite-trajectory DiCE",
            "training_lambda_P": 0,
            "evaluation": "exact population metrics at checkpoints only",
        },
        "sources": source_manifest,
        "validation": validation,
        "paired_results": paired,
        "outputs": {
            "result_directory": str(output),
            "figure_pdf": str(figure_pdf),
            "figure_png": str(figure_png),
            "paper_candidate_pdf": str(paper_pdf),
            "paper_candidate_png": str(paper_png),
        },
    }
    with (output / "experiment_result.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"OUTPUT={output}")
    print(json.dumps({"validation": validation, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
