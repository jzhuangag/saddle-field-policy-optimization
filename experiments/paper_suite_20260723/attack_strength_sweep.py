"""Attack-severity robustness sweep for the four neural Markov games.

The nominal setting is eta=1.  Only the adverse consequence in each reward
model is scaled, while transitions, policy classes, initializations, controller
parameters, and baseline learning rates remain fixed.  The minimizing player
receives the negative of the stored maximizing-player reward, so every setting
remains zero-sum by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import markov_game_suite as suite
from journal_games import (
    Game,
    cyclic_control,
    frequency_hopping,
    routing_interdiction,
    security_patrol,
)


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ENVIRONMENTS = (
    "CyclicControl",
    "FrequencyHopping",
    "RoutingInterdiction",
    "SecurityPatrol",
)
FACTORIES = {
    "CyclicControl": cyclic_control,
    "FrequencyHopping": frequency_hopping,
    "RoutingInterdiction": routing_interdiction,
    "SecurityPatrol": security_patrol,
}
DEFAULT_ETAS = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
PILOT_ETAS = (0.50, 1.00, 2.00)
BASELINE_LR = {
    "GDA": 0.03,
    "Adam-GDA": 0.001,
    "EGM": 0.03,
    "PPM-3": 0.03,
}
STYLES = {
    "QP+G": ("black", "-", "o"),
    "noG": ("tab:green", "-", "s"),
    "GDA": ("tab:orange", "--", "^"),
    "Adam-GDA": ("tab:blue", "--", "v"),
    "EGM": ("tab:purple", "-.", "D"),
    "PPM-3": ("tab:red", ":", "P"),
}


def attack_game(environment: str, eta: float) -> Game:
    """Return the eta-parameterized game, with eta=1 exactly nominal."""

    if eta <= 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and positive")
    base = FACTORIES[environment]()
    rewards = base.rewards.detach().clone()

    if environment == "CyclicControl":
        rewards = torch.clamp(rewards, min=0.0) + eta * torch.clamp(
            rewards, max=0.0
        )
    elif environment == "FrequencyHopping":
        for action in range(rewards.shape[1]):
            rewards[:, action, action] += 1.10 * (1.0 - eta)
    elif environment == "RoutingInterdiction":
        for action in range(rewards.shape[1]):
            rewards[:, action, action] += 1.20 * (1.0 - eta)
    elif environment == "SecurityPatrol":
        action_max = torch.arange(rewards.shape[1])[:, None]
        action_min = torch.arange(rewards.shape[2])[None, :]
        miss = action_max != action_min
        rewards[:, miss] *= eta
    else:
        raise KeyError(environment)

    return Game(
        name=f"{environment}-eta-{eta:.2f}",
        features=base.features.detach().clone(),
        rewards=rewards,
        transitions=base.transitions.detach().clone(),
        rho=base.rho.detach().clone(),
    )


def nominal_equivalence_errors() -> dict[str, float]:
    errors = {}
    for environment in ENVIRONMENTS:
        base = FACTORIES[environment]()
        nominal = attack_game(environment, 1.0)
        errors[environment] = max(
            float(torch.max(torch.abs(base.rewards - nominal.rewards))),
            float(torch.max(torch.abs(base.transitions - nominal.transitions))),
            float(torch.max(torch.abs(base.features - nominal.features))),
            float(torch.max(torch.abs(base.rho - nominal.rho))),
        )
    return errors


def t_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - half_width, mean + half_width


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    etas = PILOT_ETAS if args.pilot else tuple(args.etas)
    seed_count = 3 if args.pilot else args.seeds
    seeds = tuple(range(args.seed_start, args.seed_start + seed_count))
    methods = tuple(args.methods)
    environments = tuple(args.environments)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    maximum_soft_residual = 0.0
    maximum_hard_residual = 0.0

    nominal_errors = nominal_equivalence_errors()
    if max(nominal_errors.values()) > 1.0e-14:
        raise RuntimeError(f"eta=1 nominal-equivalence sentinel failed: {nominal_errors}")

    for environment in environments:
        for eta in etas:
            game = attack_game(environment, eta)
            transition_error = float(
                torch.max(torch.abs(game.transitions.sum(dim=-1) - 1.0))
            )
            if transition_error > 1.0e-12:
                raise RuntimeError(
                    f"transition sentinel failed for {environment}, eta={eta}"
                )
            print(f"ENVIRONMENT={environment} ETA={eta:.2f}", flush=True)
            for seed in seeds:
                z0 = suite.initialize(seed, "neural", game)
                energy0, gap0, soft_residual, _ = suite.raw_components(
                    z0, game, "neural"
                )
                maximum_soft_residual = max(maximum_soft_residual, soft_residual)
                normalizers = (max(energy0, 1.0e-10), max(gap0, 1.0e-10))

                for method in methods:
                    z = z0.detach().clone()
                    adam_state = None
                    for step in range(args.steps):
                        if method in ("QP+G", "noG"):
                            use_g = method == "QP+G"
                            if use_g:
                                f, g, rotation, cosine = suite.field_and_geometry(
                                    z, game, "neural"
                                )
                            else:
                                f = suite.field(z, game, "neural")
                                g = torch.zeros_like(f)
                                rotation = float("nan")
                                cosine = float("nan")
                            coefficient = suite.directional_coefficients(
                                z,
                                f,
                                g,
                                game,
                                "neural",
                                normalizers,
                                use_g,
                            )
                            beta, gamma, predicted = suite.solve_box(
                                coefficient, use_g
                            )
                            (
                                z,
                                drift,
                                backtracks,
                                beta_used,
                                gamma_used,
                                safeguard_residual,
                            ) = suite.safeguarded_step(
                                z,
                                f,
                                g,
                                beta,
                                gamma,
                                game,
                                "neural",
                                normalizers,
                            )
                            maximum_soft_residual = max(
                                maximum_soft_residual,
                                coefficient["max_soft_residual"],
                                safeguard_residual,
                            )
                            if use_g:
                                f_norm = float(torch.linalg.norm(f))
                                g_norm = float(torch.linalg.norm(g))
                                contribution = gamma_used * g_norm / max(
                                    beta_used * f_norm + gamma_used * g_norm,
                                    1.0e-15,
                                )
                                diagnostics.append(
                                    {
                                        "environment": environment,
                                        "eta": eta,
                                        "seed": seed,
                                        "step": step,
                                        "rotation": rotation,
                                        "cos_fg": cosine,
                                        "beta": beta_used,
                                        "gamma": gamma_used,
                                        "field_norm": f_norm,
                                        "curvature_norm": g_norm,
                                        "curvature_contribution": contribution,
                                        "predicted_decrease": -predicted,
                                        "realized_decrease": -drift,
                                        "backtracks": backtracks,
                                    }
                                )
                        else:
                            suite.LR = BASELINE_LR[method]
                            z, adam_state = suite.classical_step(
                                method, z, game, "neural", adam_state
                            )

                    br_max, br_min, hard_residual, hard_iterations = (
                        suite.hard_br_values(z, game, "neural")
                    )
                    regularized_gap, soft_final_residual, soft_final_iterations = (
                        suite.regularized_gap(z, game, "neural")
                    )
                    final_field = suite.field(z, game, "neural")
                    maximum_hard_residual = max(
                        maximum_hard_residual, hard_residual
                    )
                    maximum_soft_residual = max(
                        maximum_soft_residual, soft_final_residual
                    )
                    rows.append(
                        {
                            "environment": environment,
                            "eta": eta,
                            "seed": seed,
                            "method": method,
                            "steps": args.steps,
                            "final_unregularized_worst_case_return": br_min,
                            "final_unregularized_best_response_return": br_max,
                            "final_unregularized_exploitability": br_max - br_min,
                            "final_regularized_nash_gap": regularized_gap,
                            "final_regularized_field_norm": float(
                                torch.linalg.norm(final_field)
                            ),
                            "hard_br_residual": hard_residual,
                            "hard_br_iterations": hard_iterations,
                            "soft_br_residual": soft_final_residual,
                            "soft_br_iterations": soft_final_iterations,
                        }
                    )
                print(f"  seed {seed} complete", flush=True)

    summaries: list[dict] = []
    gains: list[dict] = []
    for environment in environments:
        for eta in etas:
            for method in methods:
                values = np.asarray(
                    [
                        row["final_unregularized_worst_case_return"]
                        for row in rows
                        if row["environment"] == environment
                        and row["eta"] == eta
                        and row["method"] == method
                    ],
                    dtype=float,
                )
                mean, lower, upper = t_interval(values)
                summaries.append(
                    {
                        "environment": environment,
                        "eta": eta,
                        "method": method,
                        "mean": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "seed_count": len(values),
                    }
                )
            if "QP+G" in methods and "noG" in methods:
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
                paired = np.asarray([qpg[seed] - nog[seed] for seed in seeds])
                mean, lower, upper = t_interval(paired)
                gains.append(
                    {
                        "environment": environment,
                        "eta": eta,
                        "mean_qpg_minus_nog": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "positive_mean": bool(mean > 0.0),
                        "positive_ci95": bool(lower > 0.0),
                        "positive_seed_count": int(np.sum(paired > 0.0)),
                        "seed_count": len(paired),
                    }
                )

    root = Path(__file__).resolve().parent
    mode = "pilot" if args.pilot else "full"
    environment_tag = (
        "all4"
        if environments == ENVIRONMENTS
        else "-".join(environment.lower() for environment in environments)
    )
    output = (
        root
        / "results"
        / f"attack-severity-{mode}-{environment_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)

    for filename, data in (
        ("final_returns.csv", rows),
        ("summary.csv", summaries),
        ("paired_gains.csv", gains),
        ("diagnostics.csv", diagnostics),
    ):
        if not data:
            continue
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    fig, axes = plt.subplots(
        2,
        len(environments),
        figsize=(3.9 * len(environments), 6.4),
        sharex="col",
        squeeze=False,
    )
    eta_array = np.asarray(etas, dtype=float)
    for column, environment in enumerate(environments):
        top = axes[0, column]
        bottom = axes[1, column]
        for method in methods:
            records = sorted(
                [
                    row
                    for row in summaries
                    if row["environment"] == environment
                    and row["method"] == method
                ],
                key=lambda row: row["eta"],
            )
            mean = np.asarray([row["mean"] for row in records])
            lower = np.asarray([row["ci95_lower"] for row in records])
            upper = np.asarray([row["ci95_upper"] for row in records])
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
            top.fill_between(
                eta_array, lower, upper, color=color, alpha=0.08, linewidth=0
            )
        gain_records = sorted(
            [row for row in gains if row["environment"] == environment],
            key=lambda row: row["eta"],
        )
        gain_mean = np.asarray(
            [row["mean_qpg_minus_nog"] for row in gain_records]
        )
        gain_lower = np.asarray([row["ci95_lower"] for row in gain_records])
        gain_upper = np.asarray([row["ci95_upper"] for row in gain_records])
        bottom.plot(
            eta_array,
            gain_mean,
            color="black",
            marker="o",
            markersize=4.2,
            linewidth=1.7,
        )
        bottom.fill_between(
            eta_array,
            gain_lower,
            gain_upper,
            color="black",
            alpha=0.12,
            linewidth=0,
        )
        for axis in (top, bottom):
            axis.axvline(1.0, color="0.55", linestyle="--", linewidth=1.0)
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=8.5)
        bottom.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
        top.set_title(environment, fontsize=10)
        bottom.set_xlabel(r"attack-severity multiplier $\eta$", fontsize=9)
    axes[0, 0].set_ylabel("final worst-case return", fontsize=9)
    axes[1, 0].set_ylabel(r"paired return gain: QP+G $-$ noG", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.25, w_pad=1.0)
    fig.savefig(output / "attack_strength_sweep.pdf", bbox_inches="tight")
    fig.savefig(
        output / "attack_strength_sweep.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    per_environment = {}
    for environment in environments:
        selected = [row for row in gains if row["environment"] == environment]
        per_environment[environment] = {
            "all_eta_positive_mean_gain": bool(
                selected and all(row["positive_mean"] for row in selected)
            ),
            "all_eta_positive_ci95": bool(
                selected and all(row["positive_ci95"] for row in selected)
            ),
            "minimum_mean_gain": min(
                (row["mean_qpg_minus_nog"] for row in selected), default=None
            ),
            "minimum_ci95_lower": min(
                (row["ci95_lower"] for row in selected), default=None
            ),
        }

    report = {
        "protocol": {
            "study": "attack-severity robustness",
            "mode": mode,
            "policy": "neural 4-8-3 tanh-softmax for each player",
            "environments": environments,
            "attack_severity_values": etas,
            "nominal_attack_severity": 1.0,
            "methods": methods,
            "seeds": seeds,
            "steps": args.steps,
            "metric": "final unregularized worst-case return",
            "comparison": "paired QP+G minus noG return gain",
            "baseline_learning_rates": BASELINE_LR,
            "qp_caps": [suite.BETA_MAX, suite.GAMMA_MAX],
            "discount": suite.DISCOUNT,
            "entropy_tau": suite.ENTROPY_TAU,
            "zero_sum_construction": "the minimizing-player payoff is the negative of the stored maximizing-player reward",
            "retuning_across_eta": False,
        },
        "attack_parameterization": {
            "CyclicControl": "scale only negative RPS outcomes by eta",
            "FrequencyHopping": "replace the jammed-action loss -1.10 by -1.10 eta",
            "RoutingInterdiction": "scale the additional matched-route interdiction loss 1.20 by eta",
            "SecurityPatrol": "scale the uncovered-target loss coefficient 0.55 by eta",
        },
        "nominal_equivalence_max_errors": nominal_errors,
        "maximum_nominal_equivalence_error": max(nominal_errors.values()),
        "maximum_training_soft_br_bellman_residual": maximum_soft_residual,
        "maximum_evaluation_hard_br_bellman_residual": maximum_hard_residual,
        "paired_gain_results": gains,
        "per_environment_positive_checks": per_environment,
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("POSITIVE_CHECKS=" + json.dumps(per_environment, sort_keys=True))
    print("RESULT_DIR=" + str(output), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=ENVIRONMENTS,
        default=ENVIRONMENTS,
    )
    parser.add_argument("--etas", type=float, nargs="+", default=DEFAULT_ETAS)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--methods", nargs="+", choices=suite.METHODS, default=suite.METHODS
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
