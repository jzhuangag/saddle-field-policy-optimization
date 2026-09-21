"""Analyze the RoutingInterdiction eta=2 finite-trajectory batch sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ENVIRONMENT = "RoutingInterdiction"
ETA = 2.0
METHODS = ("QP+G", "noG")
EXPECTED_SEEDS = tuple(range(5100, 5105))
EXPECTED_STEPS = tuple(range(0, 61, 10))
EXPECTED_BATCHES = (2048, 4096, 8192)
METRICS = (
    ("return_gain", "hard_br_return", 1.0),
    ("exploitability_reduction", "hard_exploitability", -1.0),
    ("regularized_gap_reduction", "regularized_gap", -1.0),
    ("field_norm_reduction", "field_norm", -1.0),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean, sem, mean - half, mean + half


def one_sided_sign_flip_p(values: np.ndarray) -> float:
    observed = float(np.mean(values))
    null_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null_means.append(float(np.mean(values * np.asarray(signs))))
    return float(np.mean(np.asarray(null_means) >= observed - 1.0e-15))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[int(index)]
        running = max(running, candidate)
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def load_source(source: Path) -> tuple[int, list[dict], list[dict], dict]:
    with (source / "protocol.json").open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    batch = int(protocol["transitions_per_update"])
    if batch not in EXPECTED_BATCHES:
        raise RuntimeError(f"unexpected transition batch {batch} in {source}")
    if int(protocol["trajectories_per_update"]) * int(protocol["horizon"]) != batch:
        raise RuntimeError(f"trajectory accounting mismatch in {source}")
    rows = [
        row
        for row in read_csv(source / "curves.csv")
        if row["environment"] == ENVIRONMENT and math.isclose(float(row["eta"]), ETA)
    ]
    diagnostics = [
        row
        for row in read_csv(source / "diagnostics.csv")
        if row["environment"] == ENVIRONMENT and math.isclose(float(row["eta"]), ETA)
    ]
    expected_curve_keys = {
        (seed, method, step)
        for seed in EXPECTED_SEEDS
        for method in METHODS
        for step in EXPECTED_STEPS
    }
    curve_keys = {
        (int(row["seed"]), row["method"], int(row["step"])) for row in rows
    }
    if curve_keys != expected_curve_keys or len(rows) != len(expected_curve_keys):
        raise RuntimeError(f"curve grid mismatch for batch {batch}")
    expected_diagnostic_keys = {
        (seed, method, step)
        for seed in EXPECTED_SEEDS
        for method in METHODS
        for step in range(1, 61)
    }
    diagnostic_keys = {
        (int(row["seed"]), row["method"], int(row["step"]))
        for row in diagnostics
    }
    if (
        diagnostic_keys != expected_diagnostic_keys
        or len(diagnostics) != len(expected_diagnostic_keys)
    ):
        raise RuntimeError(f"diagnostic grid mismatch for batch {batch}")
    for row in rows:
        if int(float(row["transition_batch"])) != batch:
            raise RuntimeError(f"curve batch mismatch for {source}")
        if int(float(row["cumulative_transitions"])) != int(row["step"]) * batch:
            raise RuntimeError(f"cumulative-transition mismatch for batch {batch}")
    for row in diagnostics:
        if int(float(row["transition_batch"])) != batch:
            raise RuntimeError(f"diagnostic batch mismatch for {source}")
        if int(float(row["transitions_used"])) != batch:
            raise RuntimeError(f"per-update transition mismatch for batch {batch}")
    numeric_curve = [
        float(value)
        for row in rows
        for key, value in row.items()
        if key not in {"environment", "method"}
    ]
    numeric_diagnostics = [
        float(value)
        for row in diagnostics
        for key, value in row.items()
        if key not in {"environment", "method"}
    ]
    if not np.all(np.isfinite(numeric_curve + numeric_diagnostics)):
        raise RuntimeError(f"nonfinite value in batch {batch}")
    return batch, rows, diagnostics, protocol


def summarize(
    curves_by_batch: dict[int, list[dict]],
    diagnostics_by_batch: dict[int, list[dict]],
) -> tuple[list[dict], list[dict]]:
    paired_rows: list[dict] = []
    seed_rows: list[dict] = []
    for batch in EXPECTED_BATCHES:
        curves = curves_by_batch[batch]
        final = {
            (int(row["seed"]), row["method"]): row
            for row in curves
            if int(row["step"]) == 60
        }
        paired_row = {
            "transition_batch": batch,
            "trajectories_per_update": batch // 16,
            "seed_count": len(EXPECTED_SEEDS),
        }
        differences_by_name: dict[str, np.ndarray] = {}
        for seed in EXPECTED_SEEDS:
            qpg = final[(seed, "QP+G")]
            nog = final[(seed, "noG")]
            seed_row = {
                "transition_batch": batch,
                "trajectories_per_update": batch // 16,
                "seed": seed,
            }
            for name, source_metric, sign in METRICS:
                difference = sign * (
                    float(qpg[source_metric]) - float(nog[source_metric])
                )
                seed_row[name] = difference
            seed_rows.append(seed_row)
        for name, _, _ in METRICS:
            values = np.asarray(
                [
                    row[name]
                    for row in seed_rows
                    if row["transition_batch"] == batch
                ],
                dtype=float,
            )
            differences_by_name[name] = values
            mean, sem, lower, upper = mean_ci(values)
            paired_row[f"{name}_mean"] = mean
            paired_row[f"{name}_sem"] = sem
            paired_row[f"{name}_ci95_lower"] = lower
            paired_row[f"{name}_ci95_upper"] = upper
            paired_row[f"{name}_positive_seed_count"] = int(np.sum(values > 0.0))
            paired_row[f"{name}_sign_flip_p_one_sided"] = one_sided_sign_flip_p(values)
        qpg_diagnostics = [
            row for row in diagnostics_by_batch[batch] if row["method"] == "QP+G"
        ]
        paired_row["gamma_activation"] = float(
            np.mean([float(row["gamma"]) > 1.0e-10 for row in qpg_diagnostics])
        )
        paired_row["positive_curvature_signal_rate"] = float(
            np.mean([float(row["d"]) > 0.0 for row in qpg_diagnostics])
        )
        for key in (
            "field_norm",
            "curvature_norm",
            "inflation",
            "predicted_decrease",
            "realized_decrease",
            "backtracks",
        ):
            paired_row[f"qpg_{key}_mean"] = float(
                np.mean([float(row[key]) for row in qpg_diagnostics])
            )
        paired_rows.append(paired_row)
    for name, _, _ in METRICS:
        raw = [float(row[f"{name}_sign_flip_p_one_sided"]) for row in paired_rows]
        adjusted = holm_adjust(raw)
        for row, value in zip(paired_rows, adjusted):
            row[f"{name}_holm_p"] = value
    return paired_rows, seed_rows


def plot_results(paired_rows: list[dict], seed_rows: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
        }
    )
    x = np.asarray([row["trajectories_per_update"] for row in paired_rows])
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.5))
    panels = (
        ("return_gain", "Worst-case-return gain"),
        ("exploitability_reduction", "Exploitability reduction"),
        ("field_norm_reduction", "Field-norm reduction"),
    )
    for axis, (metric, title) in zip(axes.flat[:3], panels):
        for seed in EXPECTED_SEEDS:
            selected = sorted(
                (row for row in seed_rows if row["seed"] == seed),
                key=lambda row: row["trajectories_per_update"],
            )
            axis.plot(
                x,
                [row[metric] for row in selected],
                color="0.75",
                linewidth=0.8,
                marker="o",
                markersize=2.8,
                zorder=1,
            )
        means = np.asarray([row[f"{metric}_mean"] for row in paired_rows])
        lower = np.asarray([row[f"{metric}_ci95_lower"] for row in paired_rows])
        upper = np.asarray([row[f"{metric}_ci95_upper"] for row in paired_rows])
        axis.errorbar(
            x,
            means,
            yerr=np.vstack((means - lower, upper - means)),
            color="black",
            linewidth=1.7,
            marker="s",
            markersize=4.5,
            capsize=3,
            label="mean and 95% CI",
            zorder=2,
        )
        axis.axhline(0.0, color="tab:red", linestyle="--", linewidth=0.9)
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [str(int(value)) for value in x])
        axis.set_xlabel("trajectories per update")
        axis.grid(alpha=0.25)
    diagnostic_axis = axes.flat[3]
    diagnostic_axis.plot(
        x,
        [row["gamma_activation"] for row in paired_rows],
        color="black",
        marker="o",
        linewidth=1.7,
        label=r"$\gamma_k>0$",
    )
    diagnostic_axis.plot(
        x,
        [row["positive_curvature_signal_rate"] for row in paired_rows],
        color="tab:blue",
        marker="s",
        linewidth=1.7,
        label=r"$d_k>0$",
    )
    diagnostic_axis.set_xscale("log", base=2)
    diagnostic_axis.set_xticks(x, [str(int(value)) for value in x])
    diagnostic_axis.set_xlabel("trajectories per update")
    diagnostic_axis.set_ylabel("fraction of updates")
    diagnostic_axis.set_ylim(0.0, 1.04)
    diagnostic_axis.set_title("Curvature-use diagnostics")
    diagnostic_axis.grid(alpha=0.25)
    diagnostic_axis.legend(frameon=False, loc="lower right")
    axes.flat[0].legend(frameon=False, loc="best")
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        label_y = 0.82 if label == "(d)" else 0.98
        axis.text(
            0.01,
            label_y,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
    figure.suptitle(r"RoutingInterdiction at attack severity $\eta=2$", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, nargs=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "output"
        / "pdf"
        / "fig_vi_d_routing_batch_sensitivity_candidate.pdf",
    )
    args = parser.parse_args()
    loaded = [load_source(source.resolve()) for source in args.sources]
    if len({batch for batch, _, _, _ in loaded}) != len(EXPECTED_BATCHES):
        raise RuntimeError("sources must contain one run for each expected batch")
    curves_by_batch = {batch: rows for batch, rows, _, _ in loaded}
    diagnostics_by_batch = {batch: rows for batch, _, rows, _ in loaded}
    protocols = {batch: protocol for batch, _, _, protocol in loaded}
    paired_rows, seed_rows = summarize(curves_by_batch, diagnostics_by_batch)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output_root / f"routing-eta2-batch-sensitivity-{timestamp}"
    output.mkdir(parents=True)
    combined_curves = [
        row for batch in EXPECTED_BATCHES for row in curves_by_batch[batch]
    ]
    combined_diagnostics = [
        row for batch in EXPECTED_BATCHES for row in diagnostics_by_batch[batch]
    ]
    write_csv(output / "curves.csv", combined_curves)
    write_csv(output / "diagnostics.csv", combined_diagnostics)
    write_csv(output / "seed_differences.csv", seed_rows)
    write_csv(output / "paired_summary.csv", paired_rows)
    figure = output / "routing_eta2_batch_sensitivity.pdf"
    plot_results(paired_rows, seed_rows, figure)
    args.paper_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure, args.paper_output)
    initial_equality_error = 0.0
    max_hard_br_residual = 0.0
    max_soft_br_residual = 0.0
    max_game_value_residual = 0.0
    for batch in EXPECTED_BATCHES:
        rows = curves_by_batch[batch]
        initial = {
            (int(row["seed"]), row["method"]): row
            for row in rows
            if int(row["step"]) == 0
        }
        for seed in EXPECTED_SEEDS:
            for _, metric, _ in METRICS:
                initial_equality_error = max(
                    initial_equality_error,
                    abs(
                        float(initial[(seed, "QP+G")][metric])
                        - float(initial[(seed, "noG")][metric])
                    ),
                )
        max_hard_br_residual = max(
            max_hard_br_residual,
            max(float(row["hard_br_residual"]) for row in rows),
        )
        max_soft_br_residual = max(
            max_soft_br_residual,
            max(float(row["soft_br_residual"]) for row in rows),
        )
        max_game_value_residual = max(
            max_game_value_residual,
            max(float(row["game_value_residual"]) for row in rows),
        )
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "verification_status": "ANALYZED",
            "version_label": "validation_v1",
        },
        "protocol": {
            "environment": ENVIRONMENT,
            "attack_severity": ETA,
            "trajectories_per_update": [batch // 16 for batch in EXPECTED_BATCHES],
            "transition_batches": EXPECTED_BATCHES,
            "seeds": EXPECTED_SEEDS,
            "methods": METHODS,
            "steps": 60,
            "horizon": 16,
            "source_protocols": protocols,
        },
        "validation": {
            "curve_rows": len(combined_curves),
            "diagnostic_rows": len(combined_diagnostics),
            "expected_curve_rows": 3 * 5 * 2 * 7,
            "expected_diagnostic_rows": 3 * 5 * 2 * 60,
            "initial_method_equality_max_error": initial_equality_error,
            "max_hard_br_residual": max_hard_br_residual,
            "max_soft_br_residual": max_soft_br_residual,
            "max_game_value_residual": max_game_value_residual,
        },
        "paired_summary": paired_rows,
        "outputs": {
            "result_directory": str(output),
            "candidate_figure": str(args.paper_output),
        },
    }
    with (output / "experiment_result.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    hashes = {
        name: sha256(output / name)
        for name in (
            "curves.csv",
            "diagnostics.csv",
            "seed_differences.csv",
            "paired_summary.csv",
        )
    }
    with (output / "hashes.json").open("w", encoding="utf-8") as handle:
        json.dump(hashes, handle, indent=2)
    print(f"OUTPUT={output}")
    print(f"FIGURE={args.paper_output}")
    print(json.dumps(report["validation"], indent=2))
    print(json.dumps(paired_rows, indent=2))


if __name__ == "__main__":
    main()
