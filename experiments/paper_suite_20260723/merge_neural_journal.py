"""Merge controller and tuned-baseline runs into the journal neural artifact.

The controller run supplies QP+G/noG curves and QP+G diagnostics.  The tuning
summary identifies one disjoint-seed final run for each fixed baseline.  This
program validates the release protocol and writes a deterministic, portable
four-environment artifact without modifying any source result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RESULTS = HERE / "results"

ENVIRONMENTS = (
    "CyclicControl",
    "FrequencyHopping",
    "RoutingInterdiction",
    "SecurityPatrol",
)
CONTROLLER_METHODS = ("QP+G", "noG")
BASELINE_METHODS = ("GDA", "Adam-GDA", "EGM", "PPM-3")
METHODS = CONTROLLER_METHODS + BASELINE_METHODS
EXPECTED_LRS = {
    "GDA": 0.03,
    "Adam-GDA": 0.001,
    "EGM": 0.03,
    "PPM-3": 0.03,
}
SEEDS = tuple(range(40, 50))
CHECKPOINTS = tuple(range(0, 61, 5))
UPDATE_STEPS = tuple(range(1, 61))

CURVE_FIELDS = (
    "environment",
    "mode",
    "seed",
    "method",
    "step",
    "current_return",
    "hard_br_return",
    "hard_exploitability",
    "regularized_gap",
    "field_norm",
    "hard_br_residual",
    "hard_br_iterations",
    "soft_br_residual",
    "soft_br_iterations",
)
DIAGNOSTIC_FIELDS = (
    "environment",
    "mode",
    "seed",
    "step",
    "rotation",
    "cos_fg",
    "d",
    "beta",
    "gamma",
    "gamma_active",
    "g_contribution",
    "predicted_decrease",
    "realized_decrease",
    "inflation",
    "backtracks",
    "max_soft_residual",
)


class MergeError(RuntimeError):
    """Raised when an input run does not instantiate the journal protocol."""


def read_json(path: Path):
    if not path.is_file():
        raise MergeError("missing JSON file: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, required_fields):
    if not path.is_file():
        raise MergeError("missing CSV file: {}".format(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in required_fields if field not in fields]
        if missing:
            raise MergeError(
                "{} is missing fields: {}".format(path, ", ".join(missing))
            )
        rows = list(reader)
    return fields, rows


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def canonical_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def source_metadata(path: Path):
    resolved = path.resolve()
    try:
        portable = resolved.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        # Never leak a machine-specific absolute path into a released artifact.
        portable = "external/{}".format(resolved.name)
    payload = canonical_bytes(resolved)
    return {
        "path": portable,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_bytes": len(payload),
    }


def resolve_recorded_directory(raw_path, tuning_summary: Path) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
        raise MergeError("recorded final-run directory does not exist: {}".format(candidate))

    candidates = (
        (PROJECT / candidate).resolve(),
        (tuning_summary.parent / candidate).resolve(),
    )
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        raise MergeError(
            "cannot resolve recorded final-run directory {!r} relative to the "
            "repository or tuning summary".format(str(raw_path))
        )
    return existing[0]


def require_equal(label, actual, expected) -> None:
    if actual != expected:
        raise MergeError("{} is {}; expected {}".format(label, actual, expected))


def require_close(label, actual, expected) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-15):
        raise MergeError("{} is {}; expected {}".format(label, actual, expected))


def validate_protocol(
    report,
    label: str,
    required_methods,
    expected_lr=None,
    allow_legacy_missing_methods=False,
) -> None:
    protocol = report.get("protocol", {})
    require_equal(label + " mode", protocol.get("mode", "neural"), "neural")
    environments = set(protocol.get("environments", ()))
    missing_environments = set(ENVIRONMENTS) - environments
    if missing_environments:
        raise MergeError(
            "{} protocol misses environments: {}".format(
                label, sorted(missing_environments)
            )
        )
    require_equal(label + " seeds", tuple(protocol.get("seeds", ())), SEEDS)
    require_equal(label + " steps", int(protocol.get("steps", -1)), 60)
    require_equal(
        label + " checkpoint_every", int(protocol.get("checkpoint_every", 5)), 5
    )
    recorded_methods = protocol.get("methods")
    if recorded_methods is None and allow_legacy_missing_methods:
        pass
    else:
        require_equal(label + " methods", tuple(recorded_methods or ()), tuple(required_methods))
    if expected_lr is not None:
        require_close(label + " fixed_lr", protocol.get("fixed_lr"), expected_lr)


def validate_tuning_protocol(tuning) -> None:
    protocol = tuning.get("protocol", {})
    environments = set(protocol.get("environments", ()))
    missing_environments = set(ENVIRONMENTS) - environments
    if missing_environments:
        raise MergeError(
            "tuning protocol misses environments: {}".format(
                sorted(missing_environments)
            )
        )
    require_equal(
        "tuning methods", tuple(protocol.get("methods", ())), BASELINE_METHODS
    )
    require_equal(
        "tuning seeds", tuple(protocol.get("tuning_seeds", ())), tuple(range(1000, 1005))
    )
    require_equal(
        "tuning final seeds", tuple(protocol.get("final_seeds", ())), SEEDS
    )
    require_equal("tuning steps", int(protocol.get("steps", -1)), 60)


def select_and_validate_curves(rows, methods, label: str):
    selected = [
        row
        for row in rows
        if row["environment"] in ENVIRONMENTS and row["method"] in methods
    ]
    grid = defaultdict(list)
    for row in selected:
        require_equal(label + " mode", row["mode"], "neural")
        try:
            seed = int(row["seed"])
            step = int(row["step"])
        except ValueError as error:
            raise MergeError("{} contains a noninteger seed or step".format(label)) from error
        grid[(row["environment"], row["method"], seed)].append(step)

    expected_keys = {
        (environment, method, seed)
        for environment in ENVIRONMENTS
        for method in methods
        for seed in SEEDS
    }
    require_equal(label + " environment/method/seed grid", set(grid), expected_keys)
    for key, steps in grid.items():
        require_equal(label + " checkpoints for {}".format(key), tuple(sorted(steps)), CHECKPOINTS)
    return selected


def select_and_validate_diagnostics(rows, label: str):
    selected = [row for row in rows if row["environment"] in ENVIRONMENTS]
    grid = defaultdict(list)
    for row in selected:
        require_equal(label + " mode", row["mode"], "neural")
        grid[(row["environment"], int(row["seed"]))].append(int(row["step"]))
    expected_keys = {
        (environment, seed) for environment in ENVIRONMENTS for seed in SEEDS
    }
    require_equal(label + " environment/seed grid", set(grid), expected_keys)
    for key, steps in grid.items():
        require_equal(label + " update steps for {}".format(key), tuple(sorted(steps)), UPDATE_STEPS)
    return selected


def validate_common_initialization(curves) -> None:
    compared_fields = [field for field in CURVE_FIELDS if field != "method"]
    exact_fields = {
        "environment",
        "mode",
        "seed",
        "step",
        "hard_br_iterations",
        "soft_br_iterations",
    }
    by_cell = defaultdict(list)
    for row in curves:
        if int(row["step"]) == 0:
            by_cell[(row["environment"], int(row["seed"]))].append(row)
    for key, rows in by_cell.items():
        require_equal("step-zero method count for {}".format(key), len(rows), len(METHODS))
        reference = {field: rows[0][field] for field in compared_fields}
        for row in rows[1:]:
            candidate = {field: row[field] for field in compared_fields}
            for field in compared_fields:
                label = "matched initialization {} for {}".format(field, key)
                if field in exact_fields:
                    require_equal(label, candidate[field], reference[field])
                elif not math.isclose(
                    float(candidate[field]),
                    float(reference[field]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise MergeError(
                        "{} is {}; expected approximately {}".format(
                            label, candidate[field], reference[field]
                        )
                    )


def summarize(curves, diagnostics):
    summaries = []
    decisions = []
    for environment in ENVIRONMENTS:
        for method in METHODS:
            final = [
                row
                for row in curves
                if row["environment"] == environment
                and row["method"] == method
                and int(row["step"]) == 60
            ]
            final.sort(key=lambda row: int(row["seed"]))
            record = {
                "environment": environment,
                "method": method,
                "seed_count": len(final),
            }
            for metric in (
                "hard_br_return",
                "hard_exploitability",
                "regularized_gap",
                "field_norm",
            ):
                values = np.asarray([float(row[metric]) for row in final], dtype=float)
                record[metric + "_mean"] = float(values.mean())
                record[metric + "_sem"] = float(
                    values.std(ddof=1) / math.sqrt(len(values))
                )
            summaries.append(record)

        qpg = sorted(
            [
                row
                for row in curves
                if row["environment"] == environment
                and row["method"] == "QP+G"
                and int(row["step"]) == 60
            ],
            key=lambda row: int(row["seed"]),
        )
        nog = sorted(
            [
                row
                for row in curves
                if row["environment"] == environment
                and row["method"] == "noG"
                and int(row["step"]) == 60
            ],
            key=lambda row: int(row["seed"]),
        )
        differences = np.asarray(
            [
                float(left["hard_br_return"]) - float(right["hard_br_return"])
                for left, right in zip(qpg, nog)
            ],
            dtype=float,
        )
        sem = float(differences.std(ddof=1) / math.sqrt(len(differences)))
        critical = float(stats.t.ppf(0.975, len(differences) - 1))
        wins = int(np.sum(differences > 0.0))
        sign_p = float(
            stats.binomtest(
                wins, len(differences), p=0.5, alternative="greater"
            ).pvalue
        )
        qpg_exploit = float(
            np.mean([float(row["hard_exploitability"]) for row in qpg])
        )
        nog_exploit = float(
            np.mean([float(row["hard_exploitability"]) for row in nog])
        )
        environment_diagnostics = [
            row for row in diagnostics if row["environment"] == environment
        ]
        confidence = [
            float(differences.mean() - critical * sem),
            float(differences.mean() + critical * sem),
        ]
        decisions.append(
            {
                "environment": environment,
                "seed_count": len(differences),
                "br_wins": wins,
                "paired_br_gain_mean": float(differences.mean()),
                "paired_br_gain_95ci": confidence,
                "one_sided_exact_sign_test_p": sign_p,
                "qpg_exploitability_mean": qpg_exploit,
                "nog_exploitability_mean": nog_exploit,
                "strict_all_seed_positive": bool(
                    wins == len(differences) and qpg_exploit < nog_exploit
                ),
                "confirmatory_positive": bool(
                    confidence[0] > 0.0
                    and sign_p < 0.05
                    and qpg_exploit < nog_exploit
                ),
                "median_rotation": float(
                    np.median(
                        [float(row["rotation"]) for row in environment_diagnostics]
                    )
                ),
                "gamma_activation": float(
                    np.mean(
                        [
                            float(row["gamma_active"])
                            for row in environment_diagnostics
                        ]
                    )
                ),
                "mean_g_contribution": float(
                    np.mean(
                        [
                            float(row["g_contribution"])
                            for row in environment_diagnostics
                        ]
                    )
                ),
                "inflation_fraction": float(
                    np.mean(
                        [
                            float(row["inflation"]) > 1.0e-10
                            for row in environment_diagnostics
                        ]
                    )
                ),
            }
        )
    return summaries, decisions


def no_absolute_windows_path(value) -> bool:
    serialized = json.dumps(value)
    return re.search(r"[A-Za-z]:[\\/]", serialized) is None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the four-environment journal neural result set."
    )
    parser.add_argument("--controller-dir", type=Path, required=True)
    parser.add_argument("--tuning-summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="destination; defaults to a new timestamped results directory",
    )
    args = parser.parse_args()

    controller = args.controller_dir.resolve()
    tuning_summary = args.tuning_summary.resolve()
    controller_report = read_json(controller / "summary.json")
    validate_protocol(
        controller_report,
        "controller",
        CONTROLLER_METHODS,
        allow_legacy_missing_methods=True,
    )

    _, controller_curve_rows = read_csv(
        controller / "curves.csv", CURVE_FIELDS
    )
    _, controller_diagnostic_rows = read_csv(
        controller / "diagnostics.csv", DIAGNOSTIC_FIELDS
    )
    controller_curves = select_and_validate_curves(
        controller_curve_rows, CONTROLLER_METHODS, "controller curves"
    )
    diagnostics = select_and_validate_diagnostics(
        controller_diagnostic_rows, "controller diagnostics"
    )

    tuning = read_json(tuning_summary)
    validate_tuning_protocol(tuning)
    selected_lrs = tuning.get("selected_lrs")
    if selected_lrs is None:
        selected_lrs = tuning.get("selected_learning_rates")
    if not isinstance(selected_lrs, dict):
        raise MergeError("tuning summary has no selected learning-rate mapping")
    require_equal("selected learning-rate methods", set(selected_lrs), set(BASELINE_METHODS))
    for method, expected_lr in EXPECTED_LRS.items():
        require_close(method + " selected learning rate", selected_lrs[method], expected_lr)

    final_runs = tuning.get("final_runs")
    if not isinstance(final_runs, dict):
        raise MergeError(
            "tuning summary has no final_runs mapping; rerun tune_fixed_baselines.py "
            "or provide its full summary.json"
        )
    require_equal("final-run methods", set(final_runs), set(BASELINE_METHODS))

    baseline_directories = {}
    baseline_curves = []
    for method in BASELINE_METHODS:
        directory = resolve_recorded_directory(final_runs[method], tuning_summary)
        baseline_directories[method] = directory
        report = read_json(directory / "summary.json")
        validate_protocol(report, method, (method,), expected_lr=selected_lrs[method])
        _, rows = read_csv(directory / "curves.csv", CURVE_FIELDS)
        baseline_curves.extend(select_and_validate_curves(rows, (method,), method + " curves"))

    environment_rank = {name: index for index, name in enumerate(ENVIRONMENTS)}
    method_rank = {name: index for index, name in enumerate(METHODS)}
    curves = controller_curves + baseline_curves
    curves.sort(
        key=lambda row: (
            environment_rank[row["environment"]],
            method_rank[row["method"]],
            int(row["seed"]),
            int(row["step"]),
        )
    )
    diagnostics.sort(
        key=lambda row: (
            environment_rank[row["environment"]],
            int(row["seed"]),
            int(row["step"]),
        )
    )
    validate_common_initialization(curves)

    summaries, decisions = summarize(curves, diagnostics)
    output = args.output_dir
    if output is None:
        output = RESULTS / (
            "neural-journal-four-" + time.strftime("%Y%m%d-%H%M%S")
        )
    elif not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if output.exists():
        raise MergeError("output directory already exists: {}".format(output))
    output.mkdir(parents=True)

    write_csv(output / "curves.csv", CURVE_FIELDS, curves)
    write_csv(output / "diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostics)
    write_csv(output / "summary.csv", tuple(summaries[0]), summaries)

    controller_protocol = controller_report.get("protocol", {})
    report = {
        "schema": "journal-neural-results/1",
        "protocol": {
            "mode": "neural",
            "environments": list(ENVIRONMENTS),
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "steps": 60,
            "checkpoint_every": 5,
            "simultaneous_updates": True,
            "warmup": "none",
            "selected_learning_rates": {
                method: float(selected_lrs[method]) for method in BASELINE_METHODS
            },
            "qp_caps": controller_protocol.get("qp_caps", [0.03, 0.03]),
            "ppm_inner": 3,
            "discount": controller_protocol.get("discount", 0.9),
            "entropy_tau": controller_protocol.get("entropy_tau", 0.03),
            "performance_merit": controller_protocol.get(
                "performance_merit",
                "exact entropy-regularized policy-space Nash gap recomputed at every stencil point",
            ),
            "direction": controller_protocol.get(
                "direction", "G=DF F unchanged"
            ),
            "fairness": (
                "fixed-baseline learning rates selected on seeds 1000--1004; "
                "reported curves use disjoint seeds 40--49"
            ),
        },
        "maximum_training_soft_br_bellman_residual": max(
            max(float(row["soft_br_residual"]) for row in curves),
            max(float(row["max_soft_residual"]) for row in diagnostics),
        ),
        "maximum_evaluation_hard_br_bellman_residual": max(
            float(row["hard_br_residual"]) for row in curves
        ),
        "summaries": summaries,
        "decisions": decisions,
        "provenance": {
            "controller": {
                "directory": source_metadata(controller / "summary.json")["path"].rsplit("/", 1)[0],
                "curves": source_metadata(controller / "curves.csv"),
                "diagnostics": source_metadata(controller / "diagnostics.csv"),
                "summary": source_metadata(controller / "summary.json"),
            },
            "tuning_summary": source_metadata(tuning_summary),
            "baseline_runs": {
                method: {
                    "directory": source_metadata(directory / "summary.json")["path"].rsplit("/", 1)[0],
                    "curves": source_metadata(directory / "curves.csv"),
                    "summary": source_metadata(directory / "summary.json"),
                }
                for method, directory in baseline_directories.items()
            },
        },
    }
    if not no_absolute_windows_path(report):
        raise MergeError("portable-output sentinel detected an absolute Windows path")
    write_json(output / "summary.json", report)

    print("CURVE_ROWS={}".format(len(curves)))
    print("DIAGNOSTIC_ROWS={}".format(len(diagnostics)))
    print("RESULT_DIR={}".format(output), flush=True)


if __name__ == "__main__":
    try:
        main()
    except MergeError as error:
        raise SystemExit("ERROR: {}".format(error)) from error
