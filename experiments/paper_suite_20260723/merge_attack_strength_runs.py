"""Merge disjoint attack-severity runs and rebuild five-seed figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attack_strength_sweep import ENVIRONMENTS, STYLES, t_interval


ETAS = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
METHODS = ("QP+G", "noG", "GDA", "Adam-GDA", "EGM", "PPM-3")
EXPECTED_SEEDS = (3000, 3001, 3002, 3003, 3004)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    source_dirs = [path.resolve() for path in args.sources]
    rows: list[dict] = []
    diagnostics: list[dict] = []
    source_reports = []
    for source in source_dirs:
        source_rows = read_csv(source / "final_returns.csv")
        for row in source_rows:
            row["eta"] = float(row["eta"])
            row["seed"] = int(row["seed"])
            row["steps"] = int(row["steps"])
            for key in (
                "final_unregularized_worst_case_return",
                "final_unregularized_best_response_return",
                "hard_br_residual",
            ):
                row[key] = float(row[key])
            row["final_unregularized_exploitability"] = (
                row["final_unregularized_best_response_return"]
                - row["final_unregularized_worst_case_return"]
            )
            rows.append(row)
        diagnostic_path = source / "diagnostics.csv"
        if diagnostic_path.exists():
            diagnostics.extend(read_csv(diagnostic_path))
        source_reports.append(
            json.loads((source / "summary.json").read_text(encoding="utf-8"))
        )

    keys = [
        (row["environment"], row["eta"], row["seed"], row["method"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate environment/eta/seed/method rows in sources")
    expected = {
        (environment, eta, seed, method)
        for environment in ENVIRONMENTS
        for eta in ETAS
        for seed in EXPECTED_SEEDS
        for method in METHODS
    }
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"merged protocol is incomplete: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )

    summaries: list[dict] = []
    paired_gains: list[dict] = []
    rankings: list[dict] = []
    for environment in ENVIRONMENTS:
        for eta in ETAS:
            method_means = {}
            for method in METHODS:
                selected = np.asarray(
                    [
                        row["final_unregularized_worst_case_return"]
                        for row in rows
                        if row["environment"] == environment
                        and row["eta"] == eta
                        and row["method"] == method
                    ],
                    dtype=float,
                )
                mean, lower, upper = t_interval(selected)
                method_means[method] = mean
                summaries.append(
                    {
                        "environment": environment,
                        "eta": eta,
                        "method": method,
                        "metric": "final_unregularized_worst_case_return",
                        "mean": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "seed_count": len(selected),
                    }
                )
            best_method = max(method_means, key=method_means.get)
            rankings.append(
                {
                    "environment": environment,
                    "eta": eta,
                    "best_method": best_method,
                    "qpg_is_best": best_method == "QP+G",
                    "qpg_mean": method_means["QP+G"],
                    "best_mean": method_means[best_method],
                }
            )

            qpg = {
                row["seed"]: row["final_unregularized_worst_case_return"]
                for row in rows
                if row["environment"] == environment
                and row["eta"] == eta
                and row["method"] == "QP+G"
            }
            nog = {
                row["seed"]: row["final_unregularized_worst_case_return"]
                for row in rows
                if row["environment"] == environment
                and row["eta"] == eta
                and row["method"] == "noG"
            }
            differences = np.asarray(
                [qpg[seed] - nog[seed] for seed in EXPECTED_SEEDS], dtype=float
            )
            mean, lower, upper = t_interval(differences)
            paired_gains.append(
                {
                    "environment": environment,
                    "eta": eta,
                    "mean_qpg_minus_nog": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "positive_mean": mean > 0.0,
                    "positive_ci95": lower > 0.0,
                    "positive_seed_count": int(np.sum(differences > 0.0)),
                    "seed_count": len(differences),
                }
            )

    exploitability_summaries: list[dict] = []
    for environment in ENVIRONMENTS:
        for eta in ETAS:
            for method in METHODS:
                selected = np.asarray(
                    [
                        row["final_unregularized_exploitability"]
                        for row in rows
                        if row["environment"] == environment
                        and row["eta"] == eta
                        and row["method"] == method
                    ],
                    dtype=float,
                )
                mean, lower, upper = t_interval(selected)
                exploitability_summaries.append(
                    {
                        "environment": environment,
                        "eta": eta,
                        "method": method,
                        "metric": "final_unregularized_exploitability",
                        "mean": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "seed_count": len(selected),
                    }
                )

    output_root = Path(__file__).resolve().parent / "results"
    output = output_root / f"attack-severity-5seed-{time.strftime('%Y%m%d-%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    rows.sort(key=lambda row: (row["environment"], row["eta"], row["seed"], row["method"]))
    write_csv(output / "final_returns.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "return_summary.csv", summaries)
    write_csv(output / "paired_return_gains.csv", paired_gains)
    write_csv(output / "exploitability_summary.csv", exploitability_summaries)

    eta_array = np.asarray(ETAS)
    fig, axes = plt.subplots(2, 4, figsize=(15.6, 6.4), sharex="col")
    for column, environment in enumerate(ENVIRONMENTS):
        top, bottom = axes[0, column], axes[1, column]
        for method in METHODS:
            selected = sorted(
                [
                    row
                    for row in summaries
                    if row["environment"] == environment
                    and row["method"] == method
                ],
                key=lambda row: row["eta"],
            )
            mean = np.asarray([row["mean"] for row in selected])
            lower = np.asarray([row["ci95_lower"] for row in selected])
            upper = np.asarray([row["ci95_upper"] for row in selected])
            color, linestyle, marker = STYLES[method]
            top.plot(
                eta_array,
                mean,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.2,
                linewidth=1.6,
                label=method,
            )
            top.fill_between(eta_array, lower, upper, color=color, alpha=0.07)
        gains = sorted(
            [row for row in paired_gains if row["environment"] == environment],
            key=lambda row: row["eta"],
        )
        mean = np.asarray([row["mean_qpg_minus_nog"] for row in gains])
        lower = np.asarray([row["ci95_lower"] for row in gains])
        upper = np.asarray([row["ci95_upper"] for row in gains])
        bottom.plot(eta_array, mean, color="black", marker="o", linewidth=1.7)
        bottom.fill_between(eta_array, lower, upper, color="black", alpha=0.12)
        bottom.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
        for axis in (top, bottom):
            axis.axvline(1.0, color="0.55", linestyle="--", linewidth=1.0)
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=8.5)
        top.set_title(environment, fontsize=10)
        bottom.set_xlabel(r"attack-severity multiplier $\eta$", fontsize=9)
    axes[0, 0].set_ylabel("final worst-case return", fontsize=9)
    axes[1, 0].set_ylabel(r"paired return gain: QP+G $-$ noG", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.25, w_pad=1.0)
    fig.savefig(output / "attack_strength_return.pdf", bbox_inches="tight")
    fig.savefig(output / "attack_strength_return.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.35), squeeze=False)
    for column, environment in enumerate(ENVIRONMENTS):
        axis = axes[0, column]
        for method in METHODS:
            selected = sorted(
                [
                    row
                    for row in exploitability_summaries
                    if row["environment"] == environment
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
                markersize=4.0,
                linewidth=1.5,
                label=method,
            )
            axis.fill_between(eta_array, lower, upper, color=color, alpha=0.07)
        axis.axvline(1.0, color="0.55", linestyle="--", linewidth=1.0)
        axis.set_title(environment, fontsize=10)
        axis.set_xlabel(r"attack-severity multiplier $\eta$", fontsize=9)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=8.5)
    axes[0, 0].set_ylabel("final unregularized Nash gap", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=1.0)
    fig.savefig(output / "attack_strength_exploitability.pdf", bbox_inches="tight")
    fig.savefig(
        output / "attack_strength_exploitability.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    positive_checks = {}
    for environment in ENVIRONMENTS:
        selected = [
            row for row in paired_gains if row["environment"] == environment
        ]
        positive_checks[environment] = {
            "all_eta_positive_mean_gain": all(
                row["positive_mean"] for row in selected
            ),
            "all_eta_positive_ci95": all(
                row["positive_ci95"] for row in selected
            ),
            "minimum_mean_gain": min(
                row["mean_qpg_minus_nog"] for row in selected
            ),
            "minimum_ci95_lower": min(row["ci95_lower"] for row in selected),
        }

    report = {
        "protocol": {
            "environments": ENVIRONMENTS,
            "attack_severity_values": ETAS,
            "seeds": EXPECTED_SEEDS,
            "methods": METHODS,
            "steps": 60,
            "policy": "neural 4-8-3 tanh-softmax for each player",
            "oracle": "exact population oracle; no finite-trajectory sampling noise",
            "primary_metric": "final unregularized worst-case return",
            "secondary_metric": "final unregularized exploitability/Nash gap",
        },
        "source_result_directories": [str(path) for path in source_dirs],
        "complete_expected_row_count": len(expected),
        "actual_row_count": len(rows),
        "qpg_best_mean_count": sum(row["qpg_is_best"] for row in rankings),
        "environment_eta_count": len(rankings),
        "all_qpg_mean_rankings_best": all(
            row["qpg_is_best"] for row in rankings
        ),
        "paired_return_positive_checks": positive_checks,
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
    print("POSITIVE_CHECKS=" + json.dumps(positive_checks, sort_keys=True))
    print(
        f"QPG_BEST={report['qpg_best_mean_count']}/"
        f"{report['environment_eta_count']}"
    )


if __name__ == "__main__":
    main()
