"""Merge five-seed endpoint diagnostics for QP+G and noG."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attack_strength_sweep import ENVIRONMENTS, STYLES, t_interval


ETAS = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
SEEDS = (3000, 3001, 3002, 3003, 3004)
METHODS = ("QP+G", "noG")
METRICS = (
    (
        "final_regularized_nash_gap",
        "regularized Nash gap",
    ),
    (
        "final_regularized_field_norm",
        "regularized field norm",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--primary-result", type=Path, required=True)
    args = parser.parse_args()

    source_dirs = [path.resolve() for path in args.sources]
    rows = []
    source_reports = []
    for source in source_dirs:
        rows.extend(read_csv(source / "final_returns.csv"))
        source_reports.append(
            json.loads((source / "summary.json").read_text(encoding="utf-8"))
        )
    for row in rows:
        row["eta"] = float(row["eta"])
        row["seed"] = int(row["seed"])
        for key in (
            "final_unregularized_worst_case_return",
            "final_regularized_nash_gap",
            "final_regularized_field_norm",
        ):
            row[key] = float(row[key])

    expected = {
        (environment, eta, seed, method)
        for environment in ENVIRONMENTS
        for eta in ETAS
        for seed in SEEDS
        for method in METHODS
    }
    actual = {
        (row["environment"], row["eta"], row["seed"], row["method"])
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("endpoint diagnostic sources are incomplete or duplicated")

    primary_rows = read_csv(args.primary_result.resolve() / "final_returns.csv")
    primary = {
        (
            row["environment"],
            float(row["eta"]),
            int(row["seed"]),
            row["method"],
        ): float(row["final_unregularized_worst_case_return"])
        for row in primary_rows
        if row["method"] in METHODS
    }
    maximum_return_reproduction_error = max(
        abs(
            row["final_unregularized_worst_case_return"]
            - primary[(row["environment"], row["eta"], row["seed"], row["method"])]
        )
        for row in rows
    )

    summaries = []
    paired = []
    for metric, _ in METRICS:
        for environment in ENVIRONMENTS:
            for eta in ETAS:
                values_by_method = {}
                for method in METHODS:
                    values = np.asarray(
                        [
                            row[metric]
                            for row in rows
                            if row["environment"] == environment
                            and row["eta"] == eta
                            and row["method"] == method
                        ],
                        dtype=float,
                    )
                    values_by_method[method] = values
                    mean, lower, upper = t_interval(values)
                    summaries.append(
                        {
                            "metric": metric,
                            "environment": environment,
                            "eta": eta,
                            "method": method,
                            "mean": mean,
                            "ci95_lower": lower,
                            "ci95_upper": upper,
                            "seed_count": len(values),
                        }
                    )
                differences = values_by_method["noG"] - values_by_method["QP+G"]
                mean, lower, upper = t_interval(differences)
                paired.append(
                    {
                        "metric": metric,
                        "environment": environment,
                        "eta": eta,
                        "mean_nog_minus_qpg": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "positive_mean": mean > 0.0,
                        "positive_ci95": lower > 0.0,
                        "positive_seed_count": int(np.sum(differences > 0.0)),
                        "seed_count": len(differences),
                    }
                )

    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"attack-severity-diagnostics-5seed-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "endpoint_metrics.csv", rows)
    write_csv(output / "metric_summary.csv", summaries)
    write_csv(output / "paired_metric_improvements.csv", paired)

    eta_array = np.asarray(ETAS)
    fig, axes = plt.subplots(2, 4, figsize=(15.6, 6.4), sharex="col")
    for row_index, (metric, ylabel) in enumerate(METRICS):
        for column, environment in enumerate(ENVIRONMENTS):
            axis = axes[row_index, column]
            for method in METHODS:
                selected = sorted(
                    [
                        row
                        for row in summaries
                        if row["metric"] == metric
                        and row["environment"] == environment
                        and row["method"] == method
                    ],
                    key=lambda row: row["eta"],
                )
                mean = np.asarray([row["mean"] for row in selected])
                lower = np.asarray([row["ci95_lower"] for row in selected])
                upper = np.asarray([row["ci95_upper"] for row in selected])
                color, linestyle, marker = STYLES[method]
                axis.plot(
                    eta_array,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.7,
                    markersize=4.2,
                    label=method,
                )
                axis.fill_between(
                    eta_array, lower, upper, color=color, alpha=0.10
                )
            axis.axvline(1.0, color="0.55", linestyle="--", linewidth=1.0)
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=8.5)
            if row_index == 0:
                axis.set_title(environment, fontsize=10)
            if row_index == 1:
                axis.set_xlabel(r"attack-severity multiplier $\eta$", fontsize=9)
            if column == 0:
                axis.set_ylabel(f"final {ylabel}", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.25, w_pad=1.0)
    fig.savefig(output / "attack_strength_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(
        output / "attack_strength_diagnostics.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    checks = {}
    for metric, _ in METRICS:
        selected = [row for row in paired if row["metric"] == metric]
        checks[metric] = {
            "positive_mean_count": sum(row["positive_mean"] for row in selected),
            "positive_ci95_count": sum(row["positive_ci95"] for row in selected),
            "total_environment_eta_count": len(selected),
            "minimum_mean_nog_minus_qpg": min(
                row["mean_nog_minus_qpg"] for row in selected
            ),
            "minimum_ci95_lower": min(row["ci95_lower"] for row in selected),
        }

    report = {
        "protocol": {
            "environments": ENVIRONMENTS,
            "attack_severity_values": ETAS,
            "seeds": SEEDS,
            "methods": METHODS,
            "steps": 60,
            "oracle": "exact population oracle; no finite-trajectory sampling noise",
            "metrics": [metric for metric, _ in METRICS],
            "paired_improvement_definition": "noG minus QP+G; positive favors QP+G because both metrics are minimized",
        },
        "source_result_directories": [str(path) for path in source_dirs],
        "primary_result_directory": str(args.primary_result.resolve()),
        "expected_endpoint_row_count": len(expected),
        "actual_endpoint_row_count": len(rows),
        "maximum_return_reproduction_error": maximum_return_reproduction_error,
        "metric_checks": checks,
        "maximum_nominal_equivalence_error": max(
            report["maximum_nominal_equivalence_error"]
            for report in source_reports
        ),
        "maximum_training_soft_br_bellman_residual": max(
            report["maximum_training_soft_br_bellman_residual"]
            for report in source_reports
        ),
        "maximum_evaluation_hard_br_bellman_residual": max(
            report["maximum_evaluation_hard_br_bellman_residual"]
            for report in source_reports
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("RESULT_DIR=" + str(output))
    print("CHECKS=" + json.dumps(checks, sort_keys=True))
    print(f"RETURN_REPRODUCTION_ERROR={maximum_return_reproduction_error:.3e}")


if __name__ == "__main__":
    main()
