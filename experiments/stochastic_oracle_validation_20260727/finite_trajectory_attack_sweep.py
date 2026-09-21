"""Finite-trajectory attack-severity robustness experiment.

This experiment keeps the Section VI-D policy architecture, DiCE oracle,
same-batch local-model stencils, safeguards, and rollout budget while varying
only the adverse reward-severity multiplier used by the population-oracle
attack sweep.  Exact dynamic-programming quantities are evaluated only at
reporting checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
import torch

import stochastic_dice_policy as dice


sys.path.insert(0, str(dice.POPULATION_DIR))
from attack_strength_sweep import (  # noqa: E402
    ENVIRONMENTS as ATTACK_ENVIRONMENTS,
    attack_game,
    nominal_equivalence_errors,
)


METHODS = ("QP+G", "noG")
DEFAULT_ENVIRONMENTS = ("SecurityPatrol", "RoutingInterdiction")
DEFAULT_ETAS = (0.5, 1.0, 1.5, 2.0)
DEFAULT_SEED_START = 5100
DEFAULT_SEEDS = 5
DEFAULT_STEPS = 60
DEFAULT_TRANSITION_BATCH = 2048


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sem_ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) <= 1:
        return mean, 0.0, mean, mean
    sem = float(array.std(ddof=1) / math.sqrt(len(array)))
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * sem)
    return mean, sem, mean - half_width, mean + half_width


def summarize(
    rows: list[dict],
    diagnostics: list[dict],
    environments: tuple[str, ...],
    etas: tuple[float, ...],
    seeds: tuple[int, ...],
    steps: int,
) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    paired: list[dict] = []
    metric_names = (
        "hard_br_return",
        "hard_exploitability",
        "regularized_gap",
        "field_norm",
    )
    for environment in environments:
        for eta in etas:
            final_by_method: dict[str, list[dict]] = {}
            for method in METHODS:
                final = sorted(
                    (
                        row
                        for row in rows
                        if row["environment"] == environment
                        and math.isclose(float(row["eta"]), eta)
                        and row["method"] == method
                        and int(row["step"]) == steps
                    ),
                    key=lambda row: int(row["seed"]),
                )
                if [int(row["seed"]) for row in final] != list(seeds):
                    raise RuntimeError(
                        f"incomplete final rows for {environment}, eta={eta}, {method}"
                    )
                final_by_method[method] = final
                summary = {
                    "environment": environment,
                    "eta": eta,
                    "method": method,
                    "seed_count": len(final),
                }
                for metric in metric_names:
                    mean, sem, lower, upper = mean_sem_ci(
                        [float(row[metric]) for row in final]
                    )
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_sem"] = sem
                    summary[f"{metric}_ci95_lower"] = lower
                    summary[f"{metric}_ci95_upper"] = upper
                summaries.append(summary)

            qpg = final_by_method["QP+G"]
            nog = final_by_method["noG"]
            differences = {
                "return_gain": np.asarray(
                    [
                        float(left["hard_br_return"])
                        - float(right["hard_br_return"])
                        for left, right in zip(qpg, nog)
                    ]
                ),
                "exploitability_reduction": np.asarray(
                    [
                        float(right["hard_exploitability"])
                        - float(left["hard_exploitability"])
                        for left, right in zip(qpg, nog)
                    ]
                ),
                "regularized_gap_reduction": np.asarray(
                    [
                        float(right["regularized_gap"])
                        - float(left["regularized_gap"])
                        for left, right in zip(qpg, nog)
                    ]
                ),
                "field_norm_reduction": np.asarray(
                    [
                        float(right["field_norm"])
                        - float(left["field_norm"])
                        for left, right in zip(qpg, nog)
                    ]
                ),
            }
            paired_row = {
                "environment": environment,
                "eta": eta,
                "seed_count": len(seeds),
            }
            for name, values in differences.items():
                mean, sem, lower, upper = mean_sem_ci(values.tolist())
                paired_row[f"{name}_mean"] = mean
                paired_row[f"{name}_sem"] = sem
                paired_row[f"{name}_ci95_lower"] = lower
                paired_row[f"{name}_ci95_upper"] = upper
                paired_row[f"{name}_positive_seed_count"] = int(
                    np.sum(values > 0.0)
                )
            selected_diagnostics = [
                row
                for row in diagnostics
                if row["environment"] == environment
                and math.isclose(float(row["eta"]), eta)
                and row["method"] == "QP+G"
            ]
            paired_row["gamma_activation"] = float(
                np.mean(
                    [float(row["gamma"]) > 1.0e-10 for row in selected_diagnostics]
                )
            )
            paired_row["positive_curvature_signal_rate"] = float(
                np.mean([float(row["d"]) > 0.0 for row in selected_diagnostics])
            )
            paired.append(paired_row)
    return summaries, paired


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    environments = tuple(args.environments)
    etas = tuple(float(value) for value in args.etas)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    steps = int(args.steps)
    transition_batch = int(args.transition_batch)
    if transition_batch % dice.HORIZON != 0:
        raise ValueError("transition batch must be divisible by the rollout horizon")
    if any(environment not in ATTACK_ENVIRONMENTS for environment in environments):
        raise ValueError(f"unsupported environment in {environments}")
    if any(eta <= 0.0 or not math.isfinite(eta) for eta in etas):
        raise ValueError("all attack-severity multipliers must be finite and positive")
    nominal_errors = nominal_equivalence_errors()
    if max(nominal_errors.values()) > 1.0e-14:
        raise RuntimeError(f"eta=1 nominal-equivalence sentinel failed: {nominal_errors}")

    eta_tag = "-".join(f"{eta:g}" for eta in etas)
    environment_tag = "-".join(environments)
    output = Path(args.output_root) / (
        f"finite-trajectory-attack-{environment_tag}-eta-{eta_tag}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output.mkdir(parents=True)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    print(
        "FINITE_TRAJECTORY_ATTACK_SWEEP\n"
        f"ENVIRONMENTS={environments} ETAS={etas} SEEDS={seeds} "
        f"STEPS={steps} TRANSITIONS_PER_UPDATE={transition_batch} "
        f"TRAJECTORIES_PER_UPDATE={transition_batch // dice.HORIZON}\n"
        f"OUTPUT_PENDING={output}",
        flush=True,
    )

    for environment_index, environment in enumerate(environments):
        for eta in etas:
            game = attack_game(environment, eta)
            transition_error = float(
                torch.max(torch.abs(game.transitions.sum(dim=-1) - 1.0))
            )
            if transition_error > 1.0e-12:
                raise RuntimeError(
                    f"transition sentinel failed for {environment}, eta={eta}"
                )
            eta_code = int(round(1000.0 * eta))
            print(f"ENVIRONMENT={environment} ETA={eta:.3f}", flush=True)
            for seed in seeds:
                z0 = dice.initialize(seed, game)
                calibration_seed = (
                    170_000_000
                    + environment_index * 10_000_000
                    + eta_code * 10_000
                    + seed * 1009
                    + transition_batch
                )
                calibration = dice.sample_batch(
                    z0,
                    game,
                    transition_batch,
                    seed=calibration_seed,
                )
                scale = dice.normalizer(z0, calibration, game)
                for method in METHODS:
                    z = z0.detach().clone()
                    cumulative_transitions = 0
                    for step in range(steps + 1):
                        if step % dice.CHECKPOINT_EVERY == 0 or step == steps:
                            exact = dice.exact_metrics(z, game)
                            rows.append(
                                {
                                    "environment": environment,
                                    "eta": eta,
                                    "seed": seed,
                                    "method": method,
                                    "step": step,
                                    "transition_batch": transition_batch,
                                    "trajectories_per_update": (
                                        transition_batch // dice.HORIZON
                                    ),
                                    "horizon": dice.HORIZON,
                                    "cumulative_transitions": cumulative_transitions,
                                    **exact,
                                }
                            )
                        if step == steps:
                            break
                        update_seed = (
                            190_000_000
                            + environment_index * 10_000_000
                            + eta_code * 10_000
                            + seed * 100_003
                            + transition_batch * 1009
                            + step * 2
                        )
                        z, diagnostic = dice.update(
                            method,
                            z,
                            game,
                            transition_batch,
                            scale,
                            update_seed,
                        )
                        cumulative_transitions += int(
                            diagnostic["transitions_used"]
                        )
                        diagnostics.append(
                            {
                                "environment": environment,
                                "eta": eta,
                                "seed": seed,
                                "method": method,
                                "step": step + 1,
                                "transition_batch": transition_batch,
                                **diagnostic,
                            }
                        )
                write_csv(output / "curves.partial.csv", rows)
                write_csv(output / "diagnostics.partial.csv", diagnostics)
                print(
                    f"  eta={eta:.3f} seed={seed} complete",
                    flush=True,
                )

    summaries, paired = summarize(
        rows, diagnostics, environments, etas, seeds, steps
    )
    write_csv(output / "curves.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_summary.csv", paired)
    protocol = {
        "study": "finite-trajectory attack-severity robustness",
        "environments": environments,
        "attack_severity_values": etas,
        "nominal_attack_severity": 1.0,
        "methods": METHODS,
        "seeds": seeds,
        "steps": steps,
        "checkpoint_every": dice.CHECKPOINT_EVERY,
        "transitions_per_update": transition_batch,
        "trajectories_per_update": transition_batch // dice.HORIZON,
        "horizon": dice.HORIZON,
        "discount": dice.DISCOUNT,
        "entropy_tau": dice.ENTROPY_TAU,
        "fixed_learning_rate": dice.LR,
        "qp_caps": [dice.BETA_MAX, dice.GAMMA_MAX],
        "policy_parameterization": "two separate 4-8-3 tanh-softmax policies",
        "oracle": (
            "finite trajectories with per-decision prefix importance ratios; "
            "at behavior parameters repeated differentiation is DiCE"
        ),
        "curvature": "same-batch autograd Jacobian-vector action G=DF F",
        "controller_merit": "normalized same-batch stochastic field energy",
        "training_best_response_calls": 0,
        "checkpoint_evaluation": (
            "exact population unregularized best responses, regularized gap, "
            "and regularized field norm"
        ),
        "nominal_equivalence_max_error": max(nominal_errors.values()),
        "elapsed_seconds": time.time() - started,
    }
    with (output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"protocol": protocol, "summaries": summaries, "paired": paired},
            handle,
            indent=2,
        )
    print(f"OUTPUT={output}", flush=True)
    print(json.dumps(paired, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environments", nargs="+", default=list(DEFAULT_ENVIRONMENTS)
    )
    parser.add_argument("--etas", type=float, nargs="+", default=list(DEFAULT_ETAS))
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--transition-batch", type=int, default=DEFAULT_TRANSITION_BATCH
    )
    parser.add_argument(
        "--output-root", type=Path, default=dice.HERE / "results"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
