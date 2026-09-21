"""Finite-trajectory attack-strength sweep for fixed-step baselines.

All methods receive exactly one batch of trajectories per joint update.  GDA
and Adam-GDA use the sampled field directly.  EGM and PPM-3 reuse the same
importance-weighted DiCE batch for their extrapolation or fixed-point field
queries, so the comparison has an equal transition budget per update.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats
import torch

import stochastic_dice_policy as dice
from finite_trajectory_attack_sweep import write_csv


import sys

sys.path.insert(0, str(dice.POPULATION_DIR))
from attack_strength_sweep import (  # noqa: E402
    ENVIRONMENTS as ATTACK_ENVIRONMENTS,
    attack_game,
    nominal_equivalence_errors,
)


METHODS = ("GDA", "Adam-GDA", "EGM", "PPM-3")
DEFAULT_ENVIRONMENTS = (
    "CyclicControl",
    "FrequencyHopping",
    "RoutingInterdiction",
    "SecurityPatrol",
)
DEFAULT_ETAS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
DEFAULT_SEED_START = 6100
DEFAULT_SEEDS = 5
DEFAULT_STEPS = 60
DEFAULT_TRANSITION_BATCH = 8192
DEFAULT_SHARED_LR = 0.03
DEFAULT_ADAM_LR = 0.001


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
    environments: tuple[str, ...],
    etas: tuple[float, ...],
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    steps: int,
) -> list[dict]:
    summaries: list[dict] = []
    metrics = (
        "hard_br_return",
        "hard_exploitability",
        "regularized_gap",
        "field_norm",
    )
    for environment in environments:
        for eta in etas:
            for method in methods:
                selected = sorted(
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
                if [int(row["seed"]) for row in selected] != list(seeds):
                    raise RuntimeError(
                        f"incomplete final rows for {environment}, eta={eta}, {method}"
                    )
                summary = {
                    "environment": environment,
                    "eta": eta,
                    "method": method,
                    "seed_count": len(selected),
                }
                for metric in metrics:
                    mean, sem, lower, upper = mean_sem_ci(
                        [float(row[metric]) for row in selected]
                    )
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_sem"] = sem
                    summary[f"{metric}_ci95_lower"] = lower
                    summary[f"{metric}_ci95_upper"] = upper
                summaries.append(summary)
    return summaries


def baseline_update(
    method: str,
    z: torch.Tensor,
    game,
    transition_batch: int,
    seed: int,
    shared_lr: float,
    adam_lr: float,
    adam_state,
):
    batch = dice.sample_batch(z, game, transition_batch, seed)
    first_field = dice.empirical_field(z, batch, game)
    field_queries = 1
    if method == "GDA":
        z_new = (z - shared_lr * first_field).detach()
    elif method == "EGM":
        predictor = (z - shared_lr * first_field).detach()
        second_field = dice.empirical_field(predictor, batch, game)
        field_queries = 2
        z_new = (z - shared_lr * second_field).detach()
    elif method == "PPM-3":
        iterate = z.detach().clone()
        for _ in range(3):
            implicit_field = dice.empirical_field(iterate, batch, game)
            iterate = (z - shared_lr * implicit_field).detach()
        field_queries = 3
        z_new = iterate
    elif method == "Adam-GDA":
        if adam_state is None:
            first_moment = torch.zeros_like(z)
            second_moment = torch.zeros_like(z)
            count = 0
        else:
            first_moment, second_moment, count = adam_state
        count += 1
        first_moment = 0.9 * first_moment + 0.1 * first_field
        second_moment = 0.999 * second_moment + 0.001 * torch.square(first_field)
        direction = (
            first_moment / (1.0 - 0.9**count)
        ) / (
            torch.sqrt(second_moment / (1.0 - 0.999**count)) + 1.0e-8
        )
        z_new = (z - adam_lr * direction).detach()
        adam_state = (first_moment, second_moment, count)
    else:
        raise ValueError(f"unsupported method: {method}")
    diagnostic = {
        "beta": adam_lr if method == "Adam-GDA" else shared_lr,
        "gamma": 0.0,
        "backtracks": 0,
        "predicted_decrease": 0.0,
        "realized_decrease": 0.0,
        "field_norm": float(torch.linalg.norm(first_field)),
        "curvature_norm": 0.0,
        "inflation": 0.0,
        "e": 0.0,
        "d": 0.0,
        "field_queries": field_queries,
        "transitions_used": batch.transitions,
    }
    return z_new, adam_state, diagnostic


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    environments = tuple(args.environments)
    etas = tuple(float(value) for value in args.etas)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    methods = tuple(args.methods)
    steps = int(args.steps)
    transition_batch = int(args.transition_batch)
    shared_lr = float(args.shared_lr)
    adam_lr = float(args.adam_lr)
    if transition_batch % dice.HORIZON != 0:
        raise ValueError("transition batch must be divisible by the rollout horizon")
    if any(environment not in ATTACK_ENVIRONMENTS for environment in environments):
        raise ValueError(f"unsupported environment in {environments}")
    if any(method not in METHODS for method in methods):
        raise ValueError(f"unsupported method in {methods}")
    if any(eta <= 0.0 or not math.isfinite(eta) for eta in etas):
        raise ValueError("all attack strengths must be finite and positive")
    if shared_lr <= 0.0 or adam_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    nominal_errors = nominal_equivalence_errors()
    if max(nominal_errors.values()) > 1.0e-14:
        raise RuntimeError(f"eta=1 nominal-equivalence sentinel failed: {nominal_errors}")

    eta_tag = "-".join(f"{eta:g}" for eta in etas)
    environment_tag = "-".join(environments)
    output = Path(args.output_root) / (
        f"finite-trajectory-baselines-{environment_tag}-eta-{eta_tag}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output.mkdir(parents=True)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    print(
        "FINITE_TRAJECTORY_BASELINE_ATTACK_SWEEP\n"
        f"ENVIRONMENTS={environments} ETAS={etas} SEEDS={seeds} METHODS={methods} "
        f"STEPS={steps} TRANSITIONS_PER_UPDATE={transition_batch} "
        f"TRAJECTORIES_PER_UPDATE={transition_batch // dice.HORIZON} "
        f"SHARED_LR={shared_lr} ADAM_LR={adam_lr}\n"
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
                for method in methods:
                    z = z0.detach().clone()
                    adam_state = None
                    cumulative_transitions = 0
                    for step in range(steps + 1):
                        if step % dice.CHECKPOINT_EVERY == 0 or step == steps:
                            rows.append(
                                {
                                    "environment": environment,
                                    "eta": eta,
                                    "seed": seed,
                                    "method": method,
                                    "step": step,
                                    "transition_batch": transition_batch,
                                    "trajectories_per_update": transition_batch // dice.HORIZON,
                                    "horizon": dice.HORIZON,
                                    "cumulative_transitions": cumulative_transitions,
                                    **dice.exact_metrics(z, game),
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
                        z, adam_state, diagnostic = baseline_update(
                            method,
                            z,
                            game,
                            transition_batch,
                            update_seed,
                            shared_lr,
                            adam_lr,
                            adam_state,
                        )
                        cumulative_transitions += int(diagnostic["transitions_used"])
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
                print(f"  eta={eta:.3f} seed={seed} complete", flush=True)

    summaries = summarize(rows, environments, etas, seeds, methods, steps)
    write_csv(output / "curves.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "summary.csv", summaries)
    protocol = {
        "study": "finite-trajectory attack-strength baseline robustness",
        "environments": environments,
        "attack_strength_values": etas,
        "methods": methods,
        "seeds": seeds,
        "steps": steps,
        "checkpoint_every": dice.CHECKPOINT_EVERY,
        "transitions_per_update": transition_batch,
        "trajectories_per_update": transition_batch // dice.HORIZON,
        "horizon": dice.HORIZON,
        "shared_learning_rate": shared_lr,
        "adam_learning_rate": adam_lr,
        "learning_rate_selection": (
            "frozen from independent population-oracle tuning seeds 1000--1004; "
            "candidate grid {0.001,0.003,0.01,0.03}"
        ),
        "same_batch_internal_queries": True,
        "equal_transition_budget_per_update": True,
        "field_queries_per_update": {
            "GDA": 1,
            "Adam-GDA": 1,
            "EGM": 2,
            "PPM-3": 3,
        },
        "oracle": (
            "finite trajectories with per-decision prefix importance ratios; "
            "EGM and PPM-3 reuse the current update batch"
        ),
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
        json.dump({"protocol": protocol, "summaries": summaries}, handle, indent=2)
    print(f"OUTPUT={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", nargs="+", default=DEFAULT_ENVIRONMENTS)
    parser.add_argument("--etas", nargs="+", type=float, default=DEFAULT_ETAS)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--transition-batch", type=int, default=DEFAULT_TRANSITION_BATCH)
    parser.add_argument("--shared-lr", type=float, default=DEFAULT_SHARED_LR)
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
