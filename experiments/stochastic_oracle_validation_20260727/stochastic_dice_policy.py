"""Trajectory-sampled stochastic-oracle validation.

The population VI-A/VI-B/VI-C experiments are imported read-only.  Training
uses finite trajectories from the current pair of neural policies.  Repeated
differentiation of a per-decision likelihood-ratio objective at the behavior
point yields the causal higher-order score terms represented by DiCE and
supplies both the stochastic game field and its Hessian-vector curvature
action.  The same trajectory batch is reused inside each QP stencil.
Exact dynamic-programming quantities are called only at reporting checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(SCRIPT_DIRECTORY / ".mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

HERE = SCRIPT_DIRECTORY
POPULATION_DIR = HERE.parent / "paper_suite_20260723"
sys.path.insert(0, str(POPULATION_DIR))

import markov_game_suite as population  # noqa: E402


DISCOUNT = 0.90
ENTROPY_TAU = 0.03
LR = 0.03
BETA_MAX = 0.03
GAMMA_MAX = 0.03
PROBE_RADIUS = 1.0e-3
PD_FLOOR = 1.0e-8
BACKTRACK_MAX = 8
HORIZON = 16
CHECKPOINT_EVERY = 10
ACTOR_DIM = population.neural_policy_dim()
TOTAL_DIM = 2 * ACTOR_DIM
METHODS = ("QP+G", "noG", "EGM")
BATCHES = (128, 512, 2048)


@dataclass(frozen=True)
class TrajectoryBatch:
    states: torch.Tensor
    protagonist_actions: torch.Tensor
    adversary_actions: torch.Tensor
    rewards: torch.Tensor
    behavior_log_joint: torch.Tensor
    horizon: int
    requested_transitions: int

    @property
    def trajectories(self) -> int:
        return int(self.states.shape[0])

    @property
    def transitions(self) -> int:
        return int(self.states.numel())


def split_policies(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return z[:ACTOR_DIM], z[ACTOR_DIM:]


def actor_policies(z: torch.Tensor, game) -> tuple[torch.Tensor, torch.Tensor]:
    protagonist, adversary = split_policies(z)
    return (
        population.neural_policy(protagonist, game.features),
        population.neural_policy(adversary, game.features),
    )


def initialize(seed: int, game) -> torch.Tensor:
    return population.initialize(seed, "neural", game)


def sign_vector() -> torch.Tensor:
    return torch.cat((-torch.ones(ACTOR_DIM), torch.ones(ACTOR_DIM)))


def categorical_choice(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    normalized = probabilities / probabilities.sum()
    return int(rng.choice(len(normalized), p=normalized))


def sample_batch(
    z: torch.Tensor,
    game,
    requested_transitions: int,
    seed: int,
    horizon: int = HORIZON,
) -> TrajectoryBatch:
    rng = np.random.default_rng(seed)
    trajectories = max(1, math.ceil(requested_transitions / horizon))
    protagonist, adversary = actor_policies(z.detach(), game)
    protagonist_np = protagonist.detach().cpu().numpy()
    adversary_np = adversary.detach().cpu().numpy()
    transitions = game.transitions.detach().cpu().numpy()
    rewards = game.rewards.detach().cpu().numpy()
    rho = game.rho.detach().cpu().numpy()
    rho = rho / rho.sum()

    states = np.empty((trajectories, horizon), dtype=np.int64)
    protagonist_actions = np.empty_like(states)
    adversary_actions = np.empty_like(states)
    sampled_rewards = np.empty((trajectories, horizon), dtype=np.float64)
    behavior_log_joint = np.empty_like(sampled_rewards)

    current_states = rng.choice(len(rho), size=trajectories, p=rho)
    for step in range(horizon):
        for trajectory in range(trajectories):
            state = int(current_states[trajectory])
            action_p = categorical_choice(protagonist_np[state], rng)
            action_q = categorical_choice(adversary_np[state], rng)
            states[trajectory, step] = state
            protagonist_actions[trajectory, step] = action_p
            adversary_actions[trajectory, step] = action_q
            sampled_rewards[trajectory, step] = rewards[state, action_p, action_q]
            behavior_log_joint[trajectory, step] = math.log(
                max(protagonist_np[state, action_p], 1.0e-300)
            ) + math.log(max(adversary_np[state, action_q], 1.0e-300))
            current_states[trajectory] = categorical_choice(
                transitions[state, action_p, action_q], rng
            )

    return TrajectoryBatch(
        states=torch.tensor(states, dtype=torch.long),
        protagonist_actions=torch.tensor(protagonist_actions, dtype=torch.long),
        adversary_actions=torch.tensor(adversary_actions, dtype=torch.long),
        rewards=torch.tensor(sampled_rewards),
        behavior_log_joint=torch.tensor(behavior_log_joint),
        horizon=horizon,
        requested_transitions=requested_transitions,
    )


def stochastic_objective(
    z: torch.Tensor,
    batch: TrajectoryBatch,
    game,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-decision importance-sampled finite-horizon return.

    At the behavior parameters, the ratios are identically one and repeated
    differentiation is the DiCE likelihood-ratio estimator.  At nearby QP
    stencil points, the explicit ratios keep the same trajectories valid
    off-policy instead of silently dropping the occupancy derivative.
    """

    protagonist, adversary = actor_policies(z, game)
    state = batch.states
    action_p = batch.protagonist_actions
    action_q = batch.adversary_actions
    log_joint = torch.log(
        protagonist[state, action_p].clamp_min(1.0e-15)
    ) + torch.log(adversary[state, action_q].clamp_min(1.0e-15))
    prefix_log_ratio = torch.cumsum(
        log_joint - batch.behavior_log_joint, dim=1
    )
    importance_ratio = torch.exp(prefix_log_ratio)

    entropy_p = population.entropy(protagonist)[state]
    entropy_q = population.entropy(adversary)[state]
    regularized_reward = (
        batch.rewards + ENTROPY_TAU * entropy_p - ENTROPY_TAU * entropy_q
    )
    discount_weights = torch.pow(
        torch.tensor(DISCOUNT, dtype=z.dtype),
        torch.arange(batch.horizon, dtype=z.dtype),
    ).reshape(1, -1)
    objective = torch.mean(
        torch.sum(discount_weights * importance_ratio * regularized_reward, dim=1)
    )
    return objective, importance_ratio


def empirical_field(
    z_value: torch.Tensor,
    batch: TrajectoryBatch,
    game,
) -> torch.Tensor:
    z = z_value.detach().clone().requires_grad_(True)
    objective, _ = stochastic_objective(z, batch, game)
    gradient = torch.autograd.grad(objective, z)[0]
    return (sign_vector() * gradient).detach()


def stochastic_oracles(
    z_value: torch.Tensor,
    batch: TrajectoryBatch,
    game,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = z_value.detach().clone().requires_grad_(True)
    objective, _ = stochastic_objective(z, batch, game)
    gradient = torch.autograd.grad(objective, z, create_graph=True)[0]
    signs = sign_vector()
    field = signs * gradient
    hessian_times_field = torch.autograd.grad(
        gradient,
        z,
        grad_outputs=field.detach(),
    )[0]
    curvature = signs * hessian_times_field
    return field.detach(), curvature.detach()


def empirical_merit(
    z: torch.Tensor,
    batch: TrajectoryBatch,
    game,
    normalizer: float,
) -> float:
    field = empirical_field(z, batch, game)
    return 0.5 * float(field @ field) / normalizer


def normalizer(z: torch.Tensor, batch: TrajectoryBatch, game) -> float:
    field = empirical_field(z, batch, game)
    return max(0.5 * float(field @ field), 1.0e-10)


def directional_coefficients(
    z: torch.Tensor,
    field: torch.Tensor,
    curvature: torch.Tensor,
    batch: TrajectoryBatch,
    game,
    scale: float,
    use_curvature: bool,
) -> dict[str, float]:
    delta_f = min(
        1.0e-2,
        PROBE_RADIUS / max(float(torch.linalg.norm(field)), 1.0e-12),
    )
    points = {
        "v0": z,
        "vfp": z + delta_f * field,
        "vfm": z - delta_f * field,
    }
    delta_g = 0.0
    if use_curvature:
        delta_g = min(
            1.0e-2,
            PROBE_RADIUS / max(float(torch.linalg.norm(curvature)), 1.0e-12),
        )
        points.update(
            {
                "vgp": z + delta_g * curvature,
                "vgm": z - delta_g * curvature,
                "vpp": z + delta_f * field + delta_g * curvature,
                "vpm": z + delta_f * field - delta_g * curvature,
                "vmp": z - delta_f * field + delta_g * curvature,
                "vmm": z - delta_f * field - delta_g * curvature,
            }
        )
    values = {
        key: empirical_merit(point, batch, game, scale)
        for key, point in points.items()
    }
    e = (values["vfp"] - values["vfm"]) / (2.0 * delta_f)
    c_raw = (
        values["vfp"] - 2.0 * values["v0"] + values["vfm"]
    ) / delta_f**2
    if not use_curvature:
        c_hat = c_raw + max(0.0, PD_FLOOR - c_raw)
        return {
            "e": e,
            "d": 0.0,
            "c_hat": c_hat,
            "a_hat": 1.0,
            "b": 0.0,
            "determinant": c_hat,
            "inflation": max(0.0, PD_FLOOR - c_raw),
        }

    d = -(values["vgp"] - values["vgm"]) / (2.0 * delta_g)
    a_raw = (
        values["vgp"] - 2.0 * values["v0"] + values["vgm"]
    ) / delta_g**2
    b = (
        values["vpp"]
        - values["vpm"]
        - values["vmp"]
        + values["vmm"]
    ) / (4.0 * delta_f * delta_g)
    hessian = np.array([[c_raw, -b], [-b, a_raw]])
    inflation = max(
        0.0, PD_FLOOR - float(np.linalg.eigvalsh(hessian)[0])
    )
    c_hat = c_raw + inflation
    a_hat = a_raw + inflation
    determinant = c_hat * a_hat - b * b
    if determinant <= 1.0e-14:
        extra = math.sqrt(abs(determinant)) + PD_FLOOR
        inflation += extra
        c_hat += extra
        a_hat += extra
        determinant = c_hat * a_hat - b * b
    return {
        "e": e,
        "d": d,
        "c_hat": c_hat,
        "a_hat": a_hat,
        "b": b,
        "determinant": determinant,
        "inflation": inflation,
    }


def safeguarded_step(
    z: torch.Tensor,
    field: torch.Tensor,
    curvature: torch.Tensor,
    beta: float,
    gamma: float,
    batch: TrajectoryBatch,
    game,
    scale: float,
) -> tuple[torch.Tensor, float, float, int, float]:
    current_merit = empirical_merit(z, batch, game, scale)
    factor = 1.0
    for backtracks in range(BACKTRACK_MAX + 1):
        beta_used = factor * beta
        gamma_used = factor * gamma
        candidate = z - beta_used * field + gamma_used * curvature
        candidate_merit = empirical_merit(candidate, batch, game, scale)
        if candidate_merit <= current_merit + 1.0e-10:
            return (
                candidate.detach(),
                beta_used,
                gamma_used,
                backtracks,
                current_merit - candidate_merit,
            )
        factor *= 0.5
    return z.detach().clone(), 0.0, 0.0, BACKTRACK_MAX + 1, 0.0


def exact_metrics(z: torch.Tensor, game) -> dict[str, float]:
    return population.metrics(z, game, "neural")


def heldout_oracle_metrics(
    z: torch.Tensor,
    game,
    transition_batch_size: int,
    seed: int,
) -> dict[str, float]:
    batch = sample_batch(z, game, transition_batch_size, seed)
    field, curvature = stochastic_oracles(z, batch, game)
    _, ratios = stochastic_objective(
        z.detach().clone().requires_grad_(True), batch, game
    )
    return {
        "heldout_stochastic_field_norm": float(torch.linalg.norm(field)),
        "heldout_stochastic_curvature_norm": float(
            torch.linalg.norm(curvature)
        ),
        "heldout_max_behavior_ratio_error": float(
            torch.max(torch.abs(ratios.detach() - 1.0))
        ),
        "heldout_transitions": batch.transitions,
    }


def update(
    method: str,
    z: torch.Tensor,
    game,
    transition_batch_size: int,
    scale: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = sample_batch(z, game, transition_batch_size, seed)
    if method == "EGM":
        first = empirical_field(z, batch, game)
        predictor = (z - LR * first).detach()
        second_batch = sample_batch(
            predictor, game, transition_batch_size, seed + 1_000_003
        )
        second = empirical_field(predictor, second_batch, game)
        return (z - LR * second).detach(), {
            "beta": LR,
            "gamma": 0.0,
            "backtracks": 0,
            "predicted_decrease": float("nan"),
            "realized_decrease": float("nan"),
            "field_norm": float(torch.linalg.norm(first)),
            "curvature_norm": float("nan"),
            "inflation": 0.0,
            "e": float("nan"),
            "d": float("nan"),
            "transitions_used": batch.transitions + second_batch.transitions,
        }

    use_curvature = method == "QP+G"
    if use_curvature:
        field, curvature = stochastic_oracles(z, batch, game)
    else:
        field = empirical_field(z, batch, game)
        curvature = torch.zeros_like(field)
    coefficients = directional_coefficients(
        z, field, curvature, batch, game, scale, use_curvature
    )
    beta, gamma, predicted = population.solve_box(
        coefficients, use_curvature
    )
    z_new, beta_used, gamma_used, backtracks, realized = safeguarded_step(
        z,
        field,
        curvature,
        beta,
        gamma,
        batch,
        game,
        scale,
    )
    return z_new, {
        "beta": beta_used,
        "gamma": gamma_used,
        "backtracks": backtracks,
        "predicted_decrease": -predicted,
        "realized_decrease": realized,
        "field_norm": float(torch.linalg.norm(field)),
        "curvature_norm": float(torch.linalg.norm(curvature)),
        "inflation": coefficients["inflation"],
        "e": coefficients["e"],
        "d": coefficients["d"],
        "transitions_used": batch.transitions,
    }


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) <= 1:
        return float(array.mean()), 0.0
    return (
        float(array.mean()),
        float(array.std(ddof=1) / math.sqrt(len(array))),
    )


def paired_interval(differences: np.ndarray) -> list[float]:
    if len(differences) <= 1:
        value = float(differences.mean())
        return [value, value]
    critical = float(stats.t.ppf(0.975, len(differences) - 1))
    sem = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    return [
        float(differences.mean() - critical * sem),
        float(differences.mean() + critical * sem),
    ]


def summarize(rows, diagnostics, batches, steps):
    summaries = []
    decisions = []
    for transition_batch_size in batches:
        for method in METHODS:
            final = sorted(
                (
                    row
                    for row in rows
                    if row["batch_size"] == transition_batch_size
                    and row["method"] == method
                    and row["step"] == steps
                ),
                key=lambda row: row["seed"],
            )
            summaries.append(
                {
                    "batch_size": transition_batch_size,
                    "method": method,
                    "hard_br_return_mean": mean_sem(
                        [row["hard_br_return"] for row in final]
                    )[0],
                    "hard_br_return_sem": mean_sem(
                        [row["hard_br_return"] for row in final]
                    )[1],
                    "hard_exploitability_mean": mean_sem(
                        [row["hard_exploitability"] for row in final]
                    )[0],
                    "population_field_norm_mean": mean_sem(
                        [row["field_norm"] for row in final]
                    )[0],
                    "heldout_stochastic_field_norm_mean": mean_sem(
                        [
                            row["heldout_stochastic_field_norm"]
                            for row in final
                        ]
                    )[0],
                }
            )

        qpg = sorted(
            (
                row
                for row in rows
                if row["batch_size"] == transition_batch_size
                and row["method"] == "QP+G"
                and row["step"] == steps
            ),
            key=lambda row: row["seed"],
        )
        nog = sorted(
            (
                row
                for row in rows
                if row["batch_size"] == transition_batch_size
                and row["method"] == "noG"
                and row["step"] == steps
            ),
            key=lambda row: row["seed"],
        )
        differences = np.asarray(
            [
                left["hard_br_return"] - right["hard_br_return"]
                for left, right in zip(qpg, nog)
            ]
        )
        selected_diagnostics = [
            row
            for row in diagnostics
            if row["batch_size"] == transition_batch_size
            and row["method"] == "QP+G"
        ]
        wins = int(np.sum(differences > 0.0))
        sign_p = float(
            stats.binomtest(
                wins, len(differences), p=0.5, alternative="greater"
            ).pvalue
        )
        decisions.append(
            {
                "batch_size": transition_batch_size,
                "paired_qpg_minus_nog_hard_br_mean": float(
                    differences.mean()
                ),
                "paired_qpg_minus_nog_hard_br_95ci": paired_interval(
                    differences
                ),
                "qpg_seed_wins": wins,
                "one_sided_exact_sign_p": sign_p,
                "qpg_exploitability_mean": float(
                    np.mean([row["hard_exploitability"] for row in qpg])
                ),
                "nog_exploitability_mean": float(
                    np.mean([row["hard_exploitability"] for row in nog])
                ),
                "gamma_activation": float(
                    np.mean(
                        [row["gamma"] > 1.0e-10 for row in selected_diagnostics]
                    )
                ),
                "positive_d_rate": float(
                    np.mean([row["d"] > 0.0 for row in selected_diagnostics])
                ),
            }
        )
    return summaries, decisions


def plot_results(rows, seeds, batches, output: Path):
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
        }
    )
    styles = {
        "QP+G": ("black", "-"),
        "noG": ("tab:green", "-"),
        "EGM": ("tab:purple", "--"),
    }
    figure, axes = plt.subplots(2, len(batches), figsize=(10.3, 5.2))
    axes = np.asarray(axes).reshape(2, len(batches))
    for column, transition_batch_size in enumerate(batches):
        for row_index, metric in enumerate(("hard_br_return", "field_norm")):
            axis = axes[row_index, column]
            for method in METHODS:
                trajectories = []
                x = None
                for seed in seeds:
                    selected = sorted(
                        (
                            row
                            for row in rows
                            if row["batch_size"] == transition_batch_size
                            and row["method"] == method
                            and row["seed"] == seed
                        ),
                        key=lambda row: row["step"],
                    )
                    trajectories.append([row[metric] for row in selected])
                    x = np.asarray([row["step"] for row in selected])
                array = np.asarray(trajectories)
                mean = array.mean(axis=0)
                sem = (
                    array.std(axis=0, ddof=1) / math.sqrt(len(array))
                    if len(array) > 1
                    else np.zeros_like(mean)
                )
                color, linestyle = styles[method]
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=method,
                )
                axis.fill_between(
                    x, mean - sem, mean + sem, color=color, alpha=0.10
                )
            direction = "higher is better" if row_index == 0 else "lower is better"
            title_metric = (
                "hard-BR return"
                if row_index == 0
                else "population field norm"
            )
            axis.set_title(
                f"batch {transition_batch_size}: {title_metric}\n({direction})"
            )
            axis.set_xlabel("simultaneous stochastic update")
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_protocol(args):
    if args.phase == "smoke":
        return (9000,), (32,), 2
    if args.phase == "screen":
        return tuple(range(3000, 3005)), BATCHES, 40
    if args.phase == "formal":
        return tuple(range(4100, 4110)), BATCHES, 60
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    batches = tuple(args.batches)
    return seeds, batches, args.steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("smoke", "screen", "formal", "custom"),
        default="smoke",
    )
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--environment", default="CyclicControl")
    args = parser.parse_args()

    seeds, batches, steps = parse_protocol(args)
    game = population.game_catalog()[args.environment]
    started = time.time()
    rows: list[dict] = []
    diagnostics: list[dict] = []
    output = HERE / "results" / (
        f"{args.phase}-{args.environment}-dice-" + time.strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True)
    print(
        f"PHASE={args.phase} ENV={game.name} SEEDS={seeds} "
        f"BATCHES={batches} STEPS={steps}\nOUTPUT_PENDING={output}",
        flush=True,
    )

    for transition_batch_size in batches:
        for seed in seeds:
            z0 = initialize(seed, game)
            calibration = sample_batch(
                z0,
                game,
                transition_batch_size,
                seed=70_000_000 + seed * 1009 + transition_batch_size,
            )
            scale = normalizer(z0, calibration, game)
            for method in METHODS:
                z = z0.detach().clone()
                cumulative_transitions = 0
                for step in range(steps + 1):
                    if step % CHECKPOINT_EVERY == 0 or step == steps:
                        exact = exact_metrics(z, game)
                        heldout = heldout_oracle_metrics(
                            z,
                            game,
                            transition_batch_size=max(
                                2048, transition_batch_size
                            ),
                            seed=80_000_000
                            + seed * 1009
                            + transition_batch_size * 31
                            + step,
                        )
                        rows.append(
                            {
                                "environment": game.name,
                                "phase": args.phase,
                                "seed": seed,
                                "batch_size": transition_batch_size,
                                "method": method,
                                "step": step,
                                "cumulative_transitions": cumulative_transitions,
                                **exact,
                                **heldout,
                            }
                        )
                    if step == steps:
                        break
                    update_seed = (
                        90_000_000
                        + seed * 100_003
                        + transition_batch_size * 1009
                        + step * 2
                    )
                    z, diagnostic = update(
                        method,
                        z,
                        game,
                        transition_batch_size,
                        scale,
                        update_seed,
                    )
                    cumulative_transitions += int(
                        diagnostic["transitions_used"]
                    )
                    diagnostics.append(
                        {
                            "environment": game.name,
                            "phase": args.phase,
                            "seed": seed,
                            "batch_size": transition_batch_size,
                            "method": method,
                            "step": step + 1,
                            **diagnostic,
                        }
                    )
            print(
                f"  batch={transition_batch_size} seed={seed} complete",
                flush=True,
            )
            write_csv(output / "curves.partial.csv", rows)
            write_csv(output / "diagnostics.partial.csv", diagnostics)

    summaries, decisions = summarize(rows, diagnostics, batches, steps)
    write_csv(output / "curves.csv", rows)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "summary.csv", summaries)
    plot_results(
        rows,
        seeds,
        batches,
        output / "stochastic_oracle_validation.pdf",
    )

    protocol = {
        "phase": args.phase,
        "environment": game.name,
        "policy_parameterization": "two separate 4-8-3 neural policies",
        "oracle": (
            "finite trajectories with per-decision importance ratios; "
            "at behavior parameters repeated differentiation is DiCE"
        ),
        "methods": METHODS,
        "seeds": seeds,
        "steps": steps,
        "checkpoint_every": CHECKPOINT_EVERY,
        "transition_batches": batches,
        "horizon": HORIZON,
        "discount": DISCOUNT,
        "entropy_tau": ENTROPY_TAU,
        "simultaneous_updates": True,
        "policy_warmup": "none",
        "fixed_lr": LR,
        "qp_caps": [BETA_MAX, GAMMA_MAX],
        "curvature": "same-batch exact autograd Hessian-vector action G=DF F",
        "controller_merit": "normalized same-batch stochastic field energy",
        "training_best_response_calls": 0,
        "evaluation": (
            "exact population hard-BR and regularized field diagnostics "
            "at checkpoints only"
        ),
        "primary_comparison": (
            "final QP+G minus noG hard-BR return at transition batch 2048"
        ),
    }
    with (output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    report = {
        "protocol": protocol,
        "summaries": summaries,
        "decisions": decisions,
        "elapsed_seconds": time.time() - started,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"OUTPUT={output}", flush=True)
    print(json.dumps(decisions, indent=2), flush=True)


if __name__ == "__main__":
    main()
