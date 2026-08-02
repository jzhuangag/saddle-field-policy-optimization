"""Integrity and paired-statistics audit for a completed formal run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


EXPECTED_SEEDS = tuple(range(4100, 4110))
EXPECTED_BATCHES = (128, 512, 2048)
EXPECTED_METHODS = ("QP+G", "noG", "EGM")
EXPECTED_STEPS = tuple(range(0, 61, 10))


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray) -> list[float]:
    critical = float(stats.t.ppf(0.975, len(values) - 1))
    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    return [
        float(values.mean() - critical * sem),
        float(values.mean() + critical * sem),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    curves_path = args.result_directory / "curves.csv"
    diagnostics_path = args.result_directory / "diagnostics.csv"
    curves = load_csv(curves_path)
    diagnostics = load_csv(diagnostics_path)

    expected_curve_keys = {
        (seed, batch, method, step)
        for seed in EXPECTED_SEEDS
        for batch in EXPECTED_BATCHES
        for method in EXPECTED_METHODS
        for step in EXPECTED_STEPS
    }
    actual_curve_keys = {
        (
            int(row["seed"]),
            int(row["batch_size"]),
            row["method"],
            int(row["step"]),
        )
        for row in curves
    }
    expected_diagnostic_keys = {
        (seed, batch, method, step)
        for seed in EXPECTED_SEEDS
        for batch in EXPECTED_BATCHES
        for method in EXPECTED_METHODS
        for step in range(1, 61)
    }
    actual_diagnostic_keys = {
        (
            int(row["seed"]),
            int(row["batch_size"]),
            row["method"],
            int(row["step"]),
        )
        for row in diagnostics
    }
    assert actual_curve_keys == expected_curve_keys
    assert actual_diagnostic_keys == expected_diagnostic_keys

    numeric_curve_columns = (
        "hard_br_return",
        "hard_exploitability",
        "field_norm",
        "hard_br_residual",
        "soft_br_residual",
        "heldout_stochastic_field_norm",
        "heldout_stochastic_curvature_norm",
        "heldout_max_behavior_ratio_error",
    )
    assert all(
        np.isfinite(float(row[column]))
        for row in curves
        for column in numeric_curve_columns
    )
    for seed in EXPECTED_SEEDS:
        for batch in EXPECTED_BATCHES:
            initial = [
                row
                for row in curves
                if int(row["seed"]) == seed
                and int(row["batch_size"]) == batch
                and int(row["step"]) == 0
            ]
            reference = initial[0]
            for row in initial[1:]:
                for metric in (
                    "hard_br_return",
                    "hard_exploitability",
                    "field_norm",
                ):
                    assert float(row[metric]) == float(reference[metric])

    report = {
        "integrity": {
            "curve_rows": len(curves),
            "diagnostic_rows": len(diagnostics),
            "complete_cartesian_keys": True,
            "finite_curve_metrics": True,
            "paired_initial_conditions_exact": True,
            "max_hard_br_residual": max(
                float(row["hard_br_residual"]) for row in curves
            ),
            "max_soft_br_residual": max(
                float(row["soft_br_residual"]) for row in curves
            ),
            "max_behavior_ratio_error": max(
                float(row["heldout_max_behavior_ratio_error"])
                for row in curves
            ),
            "curves_sha256": sha256(curves_path),
            "diagnostics_sha256": sha256(diagnostics_path),
        },
        "batch_results": [],
    }
    for batch in EXPECTED_BATCHES:
        final = [
            row
            for row in curves
            if int(row["batch_size"]) == batch and int(row["step"]) == 60
        ]
        methods = {}
        by_method = {}
        for method in EXPECTED_METHODS:
            rows = sorted(
                (row for row in final if row["method"] == method),
                key=lambda row: int(row["seed"]),
            )
            by_method[method] = rows
            methods[method] = {
                "hard_br_return_mean": float(
                    np.mean([float(row["hard_br_return"]) for row in rows])
                ),
                "hard_br_return_sem": float(
                    np.std(
                        [float(row["hard_br_return"]) for row in rows], ddof=1
                    )
                    / math.sqrt(len(rows))
                ),
                "hard_exploitability_mean": float(
                    np.mean(
                        [float(row["hard_exploitability"]) for row in rows]
                    )
                ),
                "population_field_norm_mean": float(
                    np.mean([float(row["field_norm"]) for row in rows])
                ),
            }
        comparisons = {}
        for comparator in ("noG", "EGM"):
            differences = np.asarray(
                [
                    float(left["hard_br_return"])
                    - float(right["hard_br_return"])
                    for left, right in zip(
                        by_method["QP+G"], by_method[comparator]
                    )
                ]
            )
            wins = int(np.sum(differences > 0.0))
            comparisons[f"QP+G_minus_{comparator}"] = {
                "paired_mean": float(differences.mean()),
                "paired_95_student_t_interval": interval(differences),
                "seed_wins": wins,
                "one_sided_exact_sign_p": float(
                    stats.binomtest(
                        wins,
                        len(differences),
                        p=0.5,
                        alternative="greater",
                    ).pvalue
                ),
            }
        qpg_diagnostics = [
            row
            for row in diagnostics
            if int(row["batch_size"]) == batch and row["method"] == "QP+G"
        ]
        report["batch_results"].append(
            {
                "batch_size": batch,
                "methods": methods,
                "comparisons": comparisons,
                "gamma_activation": float(
                    np.mean(
                        [float(row["gamma"]) > 1.0e-10 for row in qpg_diagnostics]
                    )
                ),
                "positive_d_rate": float(
                    np.mean(
                        [float(row["d"]) > 0.0 for row in qpg_diagnostics]
                    )
                ),
            }
        )
    with (args.result_directory / "formal_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
