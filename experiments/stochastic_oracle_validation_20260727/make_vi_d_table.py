"""Recompute the Section VI-D final-checkpoint table from paired seed data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SOURCE_CSV = (
    SCRIPT_DIR
    / "results"
    / "formal-CyclicControl-dice-20260727-124224"
    / "curves.csv"
)
OUTPUT_JSON = (
    PROJECT_ROOT
    / "output"
    / "data"
    / "vi_d_finite_trajectory_table.json"
)

METHODS = ("QP+G", "noG")
EXPECTED_SEEDS = list(range(4100, 4110))
T_CRITICAL_975_DF9 = 2.2621571627409915

METRICS = {
    "hard_br_return": {
        "label": "Worst-case return",
        "orientation": "QP+G minus noG",
        "higher_is_better": True,
    },
    "hard_exploitability": {
        "label": "Exploitability",
        "orientation": "noG minus QP+G",
        "higher_is_better": False,
    },
    "field_norm": {
        "label": "Population field norm",
        "orientation": "noG minus QP+G",
        "higher_is_better": False,
    },
}

SENTINELS = {
    "hard_br_return": {
        "QP+G_mean": -0.2033669774,
        "QP+G_se": 0.03232803896,
        "noG_mean": -0.5455649126,
        "noG_se": 0.07784823522,
        "paired_mean": 0.3421979352,
        "ci_low": 0.1230213302,
        "ci_high": 0.5613745402,
    },
    "hard_exploitability": {
        "QP+G_mean": 0.4286838672,
        "QP+G_se": 0.046539,
        "noG_mean": 0.9317828720,
        "noG_se": 0.095814,
        "paired_mean": 0.5030990048,
        "ci_low": 0.2743467067,
        "ci_high": 0.7318513030,
    },
    "field_norm": {
        "QP+G_mean": 0.1694424004,
        "QP+G_se": 0.025300,
        "noG_mean": 0.3768522258,
        "noG_se": 0.036561,
        "paired_mean": 0.2074098254,
        "ci_low": 0.1341881133,
        "ci_high": 0.2806315376,
    },
}


def sample_summary(values: List[float]) -> Dict[str, float]:
    """Return the sample mean, sample standard deviation, and standard error."""
    n = len(values)
    sample_std = statistics.stdev(values)
    return {
        "n": n,
        "mean": statistics.mean(values),
        "sample_std": sample_std,
        "standard_error": sample_std / math.sqrt(n),
    }


def paired_summary(values: List[float]) -> Dict[str, float]:
    """Return a two-sided 95% paired Student-t interval with nine degrees of freedom."""
    summary = sample_summary(values)
    margin = T_CRITICAL_975_DF9 * summary["standard_error"]
    return {
        **summary,
        "degrees_of_freedom": 9,
        "t_critical_0.975": T_CRITICAL_975_DF9,
        "confidence_level": 0.95,
        "ci_low": summary["mean"] - margin,
        "ci_high": summary["mean"] + margin,
    }


def assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    """Fail before manuscript use when a recomputed value misses its sentinel."""
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"{name}: recomputed {actual:.16g}, expected {expected:.16g}, "
            f"absolute tolerance {tolerance:.1e}"
        )


def load_filtered_rows() -> List[Dict[str, str]]:
    """Load only the prespecified CyclicControl final-checkpoint comparison."""
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "environment",
            "phase",
            "batch_size",
            "step",
            "method",
            "seed",
            *METRICS.keys(),
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise AssertionError(f"Missing required CSV columns: {sorted(missing)}")
        return [
            row
            for row in reader
            if row["environment"] == "CyclicControl"
            and row["phase"] == "formal"
            and int(float(row["batch_size"])) == 2048
            and int(float(row["step"])) == 60
            and row["method"] in METHODS
        ]


def main() -> None:
    rows = load_filtered_rows()
    if len(rows) != 20:
        raise AssertionError(f"Expected 20 filtered rows, found {len(rows)}")

    indexed: Dict[str, Dict[int, Dict[str, str]]] = {method: {} for method in METHODS}
    for row in rows:
        method = row["method"]
        seed = int(row["seed"])
        if seed in indexed[method]:
            raise AssertionError(f"Duplicate row for method={method}, seed={seed}")
        indexed[method][seed] = row

    seed_sets = {method: sorted(indexed[method]) for method in METHODS}
    for method in METHODS:
        if seed_sets[method] != EXPECTED_SEEDS:
            raise AssertionError(
                f"{method} seeds are {seed_sets[method]}, expected {EXPECTED_SEEDS}"
            )
    if seed_sets["QP+G"] != seed_sets["noG"]:
        raise AssertionError("QP+G and noG seed sets differ")

    statistics_output: Dict[str, object] = {}
    sentinel_checks: Dict[str, object] = {}
    latex_rows: Dict[str, object] = {}

    for metric, metadata in METRICS.items():
        by_method = {
            method: [float(indexed[method][seed][metric]) for seed in EXPECTED_SEEDS]
            for method in METHODS
        }
        method_summaries = {
            method: sample_summary(by_method[method]) for method in METHODS
        }
        if metadata["higher_is_better"]:
            paired_values = [
                indexed_qp - indexed_nog
                for indexed_qp, indexed_nog in zip(
                    by_method["QP+G"], by_method["noG"]
                )
            ]
        else:
            paired_values = [
                indexed_nog - indexed_qp
                for indexed_qp, indexed_nog in zip(
                    by_method["QP+G"], by_method["noG"]
                )
            ]
        paired = paired_summary(paired_values)

        statistics_output[metric] = {
            **metadata,
            "methods": method_summaries,
            "paired_improvement": paired,
            "per_seed": [
                {
                    "seed": seed,
                    "QP+G": by_method["QP+G"][index],
                    "noG": by_method["noG"][index],
                    "paired_improvement": paired_values[index],
                }
                for index, seed in enumerate(EXPECTED_SEEDS)
            ],
        }

        expected = SENTINELS[metric]
        actual_values = {
            "QP+G_mean": method_summaries["QP+G"]["mean"],
            "QP+G_se": method_summaries["QP+G"]["standard_error"],
            "noG_mean": method_summaries["noG"]["mean"],
            "noG_se": method_summaries["noG"]["standard_error"],
            "paired_mean": paired["mean"],
            "ci_low": paired["ci_low"],
            "ci_high": paired["ci_high"],
        }
        metric_checks = {}
        for key, actual in actual_values.items():
            tolerance = 1e-6 if key.endswith("_se") else 5e-10
            assert_close(f"{metric}.{key}", actual, expected[key], tolerance)
            metric_checks[key] = {
                "actual": actual,
                "expected": expected[key],
                "absolute_tolerance": tolerance,
                "passed": True,
            }
        sentinel_checks[metric] = metric_checks

        qp = method_summaries["QP+G"]
        nog = method_summaries["noG"]
        latex_rows[metric] = {
            "label": metadata["label"],
            "QP+G_display": f"{qp['mean']:.3f} ± {qp['standard_error']:.3f}",
            "noG_display": f"{nog['mean']:.3f} ± {nog['standard_error']:.3f}",
            "paired_display": (
                f"{paired['mean']:.3f} "
                f"[{paired['ci_low']:.3f}, {paired['ci_high']:.3f}]"
            ),
            "QP+G_latex": f"${qp['mean']:.3f}\\pm {qp['standard_error']:.3f}$",
            "noG_latex": f"${nog['mean']:.3f}\\pm {nog['standard_error']:.3f}$",
            "paired_latex": (
                f"${paired['mean']:.3f}$ "
                f"$[{paired['ci_low']:.3f},{paired['ci_high']:.3f}]$"
            ),
        }

    exploit_qp = statistics_output["hard_exploitability"]["methods"]["QP+G"]["mean"]
    exploit_nog = statistics_output["hard_exploitability"]["methods"]["noG"]["mean"]
    relative_reduction = 1.0 - exploit_qp / exploit_nog
    assert_close(
        "relative_exploitability_reduction",
        relative_reduction,
        0.5399315869914293,
        5e-12,
    )

    source_payload = SOURCE_CSV.read_bytes().replace(b"\r\n", b"\n").replace(
        b"\r", b"\n"
    )
    source_hash = hashlib.sha256(source_payload).hexdigest()
    output = {
        "schema": "journal-vi-d-table/1",
        "source": {
            "path": str(SOURCE_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": source_hash,
            "canonical_bytes": len(source_payload),
        },
        "filter": {
            "environment": "CyclicControl",
            "phase": "formal",
            "batch_size": 2048,
            "step": 60,
            "methods": list(METHODS),
        },
        "validation": {
            "filtered_row_count": len(rows),
            "rows_per_method": {method: len(indexed[method]) for method in METHODS},
            "seeds": EXPECTED_SEEDS,
            "same_seed_set": True,
            "one_row_per_method_seed": True,
            "excluded_batch_sizes": [128, 512],
            "excluded_methods": ["EGM"],
        },
        "statistics": statistics_output,
        "relative_exploitability_reduction": {
            "value": relative_reduction,
            "percent": 100.0 * relative_reduction,
            "definition": "1 - mean(QP+G) / mean(noG)",
        },
        "latex_three_decimal_rows": latex_rows,
        "sentinel_checks": sentinel_checks,
        "all_assertions_passed": True,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT_JSON}")
    for metric in METRICS:
        row = latex_rows[metric]
        print(
            f"{row['label']}: {row['QP+G_display']} | {row['noG_display']} | "
            f"{row['paired_display']}"
        )
    print(f"Exploitability relative reduction: {100.0 * relative_reduction:.1f}%")


if __name__ == "__main__":
    main()
