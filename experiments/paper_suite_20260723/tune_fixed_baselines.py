"""Independent learning-rate selection and final evaluation for fixed baselines.

The tuning seeds (1000--1004) are disjoint from both the earlier candidate
screening and the final reporting seeds (40--49).  One global learning rate per
method is selected across all four reported neural Markov games, which avoids
environment-specific test-set tuning.  QP+G and noG remain capped at 0.03 and
are not retuned here.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SUITE = HERE / "markov_game_suite.py"
RESULTS = HERE / "results"
ENVIRONMENTS = (
    "CyclicControl",
    "FrequencyHopping",
    "RoutingInterdiction",
    "SecurityPatrol",
)
METHODS = ("GDA", "Adam-GDA", "EGM", "PPM-3")
LR_GRID = (0.001, 0.003, 0.01, 0.03)
TUNING_SEEDS = tuple(range(1000, 1005))
FINAL_SEEDS = tuple(range(40, 50))
STEPS = 60


def repository_relative(path: Path) -> str:
    """Return a portable POSIX path rooted at the repository."""

    try:
        return path.resolve().relative_to(PROJECT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(
            "result directory is outside the repository: {}".format(path)
        ) from error


def run_suite(lr: float, methods: tuple[str, ...], seed_start: int, seed_count: int, label: str):
    command = [
        sys.executable,
        str(SUITE),
        "--mode",
        "neural",
        "--environments",
        *ENVIRONMENTS,
        "--seeds",
        str(seed_count),
        "--seed-start",
        str(seed_start),
        "--steps",
        str(STEPS),
        "--fixed-lr",
        str(lr),
        "--methods",
        *methods,
    ]
    log_path = HERE / f"{label}.log"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=HERE.parent.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"{label} failed; inspect {log_path}")
    result_line = next(
        line for line in reversed(log_path.read_text(encoding="utf-8").splitlines())
        if line.startswith("RESULT_DIR=")
    )
    return Path(result_line.split("=", 1)[1])


def read_rows(result_dir: Path):
    with (result_dir / "curves.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tuning_score(rows, method: str):
    gains = []
    exploit_ratios = []
    for environment in ENVIRONMENTS:
        for seed in TUNING_SEEDS:
            selected = [
                row for row in rows
                if row["environment"] == environment
                and row["method"] == method
                and int(row["seed"]) == seed
            ]
            selected.sort(key=lambda row: int(row["step"]))
            initial, final = selected[0], selected[-1]
            scale = max(float(initial["hard_exploitability"]), 1.0e-8)
            gains.append(
                (float(final["hard_br_return"]) - float(initial["hard_br_return"])) / scale
            )
            exploit_ratios.append(float(final["hard_exploitability"]) / scale)
    return {
        "mean_normalized_br_gain": sum(gains) / len(gains),
        "mean_normalized_exploitability": sum(exploit_ratios) / len(exploit_ratios),
    }


def main():
    started = time.time()
    tuning_runs = {}
    tuning_scores = {method: {} for method in METHODS}
    for lr in LR_GRID:
        label = f"tune_neural_lr_{lr:g}".replace(".", "p")
        result_dir = run_suite(lr, METHODS, TUNING_SEEDS[0], len(TUNING_SEEDS), label)
        tuning_runs[str(lr)] = repository_relative(result_dir)
        rows = read_rows(result_dir)
        for method in METHODS:
            tuning_scores[method][str(lr)] = tuning_score(rows, method)

    selected_lrs = {}
    for method in METHODS:
        selected_lrs[method] = max(
            LR_GRID,
            key=lambda lr: (
                tuning_scores[method][str(lr)]["mean_normalized_br_gain"],
                -tuning_scores[method][str(lr)]["mean_normalized_exploitability"],
                -lr,
            ),
        )

    final_runs = {}
    final_summaries = {}
    minimum_bridge_slack = math.inf
    for method in METHODS:
        lr = selected_lrs[method]
        label = f"final_neural_{method.replace('-', '_')}_lr_{lr:g}".replace(".", "p")
        result_dir = run_suite(lr, (method,), FINAL_SEEDS[0], len(FINAL_SEEDS), label)
        final_runs[method] = repository_relative(result_dir)
        report = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        final_summaries[method] = report["summaries"]
        minimum_bridge_slack = min(
            minimum_bridge_slack, report["minimum_performance_bridge_slack"]
        )

    report = {
        "protocol": {
            "environments": ENVIRONMENTS,
            "methods": METHODS,
            "lr_grid": LR_GRID,
            "selection": (
                "one global LR per method maximizing mean final normalized hard-BR "
                "gain across all environment-by-tuning-seed cells; normalized by "
                "initial hard exploitability, with lower normalized final "
                "exploitability and then smaller LR as tie-breakers"
            ),
            "tuning_seeds": TUNING_SEEDS,
            "final_seeds": FINAL_SEEDS,
            "steps": STEPS,
            "simultaneous_updates": True,
            "warmup": "none",
        },
        "selected_lrs": selected_lrs,
        "tuning_scores": tuning_scores,
        "tuning_runs": tuning_runs,
        "final_runs": final_runs,
        "final_summaries": final_summaries,
        "minimum_performance_bridge_slack_across_final_baseline_runs": minimum_bridge_slack,
        "elapsed_seconds": time.time() - started,
    }
    output = RESULTS / ("tuned-baselines-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    with (output / "summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print("SELECTED_LRS=" + json.dumps(selected_lrs, sort_keys=True), flush=True)
    print("RESULT_DIR=" + str(output), flush=True)


if __name__ == "__main__":
    main()
