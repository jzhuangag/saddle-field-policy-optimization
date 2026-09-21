"""Merge disjoint finite-trajectory attack-severity shards and plot results."""

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

from finite_trajectory_attack_sweep import (
    DEFAULT_ENVIRONMENTS,
    DEFAULT_ETAS,
    DEFAULT_SEED_START,
    DEFAULT_SEEDS,
    DEFAULT_STEPS,
    METHODS,
    summarize,
    write_csv,
)


EXPECTED_SEEDS = tuple(
    range(DEFAULT_SEED_START, DEFAULT_SEED_START + DEFAULT_SEEDS)
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def convert_rows(rows: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for source in rows:
        row = dict(source)
        for key in row:
            if key in {"environment", "method"}:
                continue
            if key in {
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
            }:
                row[key] = int(float(row[key]))
            else:
                row[key] = float(row[key])
        converted.append(row)
    return converted


def validate_grid(rows: list[dict], diagnostics: list[dict]) -> None:
    environments = tuple(DEFAULT_ENVIRONMENTS)
    etas = tuple(DEFAULT_ETAS)
    checkpoints = tuple(range(0, DEFAULT_STEPS + 1, 10))
    expected_rows = {
        (environment, eta, seed, method, step)
        for environment in environments
        for eta in etas
        for seed in EXPECTED_SEEDS
        for method in METHODS
        for step in checkpoints
    }
    actual_rows = {
        (
            row["environment"],
            float(row["eta"]),
            int(row["seed"]),
            row["method"],
            int(row["step"]),
        )
        for row in rows
    }
    if actual_rows != expected_rows or len(rows) != len(expected_rows):
        missing = sorted(expected_rows - actual_rows)[:5]
        extra = sorted(actual_rows - expected_rows)[:5]
        raise RuntimeError(f"curve grid mismatch; missing={missing}, extra={extra}")
    expected_diagnostics = {
        (environment, eta, seed, method, step)
        for environment in environments
        for eta in etas
        for seed in EXPECTED_SEEDS
        for method in METHODS
        for step in range(1, DEFAULT_STEPS + 1)
    }
    actual_diagnostics = {
        (
            row["environment"],
            float(row["eta"]),
            int(row["seed"]),
            row["method"],
            int(row["step"]),
        )
        for row in diagnostics
    }
    if (
        actual_diagnostics != expected_diagnostics
        or len(diagnostics) != len(expected_diagnostics)
    ):
        missing = sorted(expected_diagnostics - actual_diagnostics)[:5]
        extra = sorted(actual_diagnostics - expected_diagnostics)[:5]
        raise RuntimeError(
            f"diagnostic grid mismatch; missing={missing}, extra={extra}"
        )
    numeric_values = [
        float(value)
        for row in rows
        for key, value in row.items()
        if key not in {"environment", "method"}
    ]
    if not np.all(np.isfinite(numeric_values)):
        raise RuntimeError("nonfinite value detected in merged curves")
    for row in rows:
        expected = int(row["step"]) * int(row["transition_batch"])
        if int(row["cumulative_transitions"]) != expected:
            raise RuntimeError(
                "cumulative-transition accounting mismatch for "
                f"{row['environment']}, eta={row['eta']}, seed={row['seed']}, "
                f"method={row['method']}, step={row['step']}"
            )


def plot_summary(
    summaries: list[dict], output_pdf: Path, output_png: Path
) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    styles = {
        "QP+G": ("black", "-", "o"),
        "noG": ("tab:green", "-", "s"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.7), sharex="col")
    metrics = (
        ("hard_br_return", "Worst-case return", True),
        ("hard_exploitability", "Exploitability", False),
    )
    for column, environment in enumerate(DEFAULT_ENVIRONMENTS):
        for row_index, (metric, ylabel, higher) in enumerate(metrics):
            axis = axes[row_index, column]
            for method in METHODS:
                selected = sorted(
                    (
                        row
                        for row in summaries
                        if row["environment"] == environment
                        and row["method"] == method
                    ),
                    key=lambda row: float(row["eta"]),
                )
                x = np.asarray([float(row["eta"]) for row in selected])
                mean = np.asarray(
                    [float(row[f"{metric}_mean"]) for row in selected]
                )
                lower = np.asarray(
                    [float(row[f"{metric}_ci95_lower"]) for row in selected]
                )
                upper = np.asarray(
                    [float(row[f"{metric}_ci95_upper"]) for row in selected]
                )
                color, linestyle, marker = styles[method]
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.8,
                    markersize=4.5,
                    label=method,
                )
                axis.fill_between(x, lower, upper, color=color, alpha=0.12)
            axis.axvline(1.0, color="0.45", linestyle="--", linewidth=0.9)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(environment)
            if row_index == 1:
                axis.set_xlabel(r"attack-severity multiplier $\eta$")
            direction = "higher is better" if higher else "lower is better"
            axis.text(
                0.02,
                0.04 if row_index == 0 else 0.92,
                direction,
                transform=axis.transAxes,
                fontsize=7.5,
                color="0.35",
                va="bottom" if row_index == 0 else "top",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(figure)


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

    rows: list[dict] = []
    diagnostics: list[dict] = []
    protocols: list[dict] = []
    for source in args.sources:
        rows.extend(convert_rows(read_csv(source / "curves.csv")))
        diagnostics.extend(convert_rows(read_csv(source / "diagnostics.csv")))
        with (source / "protocol.json").open(encoding="utf-8") as handle:
            protocols.append(json.load(handle))
    validate_grid(rows, diagnostics)
    summaries, paired = summarize(
        rows,
        diagnostics,
        tuple(DEFAULT_ENVIRONMENTS),
        tuple(DEFAULT_ETAS),
        EXPECTED_SEEDS,
        DEFAULT_STEPS,
    )
    output = args.output_root / (
        "finite-trajectory-attack-5seed-" + time.strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True)
    write_csv(output / "curves.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_summary.csv", paired)
    figure_pdf = output / "finite_trajectory_attack_robustness.pdf"
    figure_png = output / "finite_trajectory_attack_robustness.png"
    plot_summary(summaries, figure_pdf, figure_png)
    args.paper_output_root.mkdir(parents=True, exist_ok=True)
    paper_pdf = args.paper_output_root / "fig_vi_d_attack_robustness_candidate.pdf"
    paper_png = args.paper_output_root / "fig_vi_d_attack_robustness_candidate.png"
    shutil.copy2(figure_pdf, paper_pdf)
    shutil.copy2(figure_png, paper_png)
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "protocol": {
            "environments": DEFAULT_ENVIRONMENTS,
            "attack_severity_values": DEFAULT_ETAS,
            "methods": METHODS,
            "seeds": EXPECTED_SEEDS,
            "steps": DEFAULT_STEPS,
            "transitions_per_update": 2048,
            "trajectories_per_update": 128,
            "horizon": 16,
            "source_protocols": protocols,
        },
        "row_counts": {
            "curves": len(rows),
            "diagnostics": len(diagnostics),
        },
        "summaries": summaries,
        "paired": paired,
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
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
