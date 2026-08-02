"""Numerical sentinels for the trajectory likelihood-ratio oracle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import stochastic_dice_policy as experiment


def finite_horizon_population_oracles(z_value, game):
    z = z_value.detach().clone().requires_grad_(True)
    protagonist, adversary = experiment.actor_policies(z, game)
    reward = torch.einsum(
        "sa,sab,sb->s", protagonist, game.rewards, adversary
    )
    reward = (
        reward
        + experiment.ENTROPY_TAU * experiment.population.entropy(protagonist)
        - experiment.ENTROPY_TAU * experiment.population.entropy(adversary)
    )
    transition = torch.einsum(
        "sa,sabn,sb->sn", protagonist, game.transitions, adversary
    )
    value = torch.zeros_like(game.rho)
    for _ in range(experiment.HORIZON):
        value = reward + experiment.DISCOUNT * (transition @ value)
    objective = game.rho @ value
    gradient = torch.autograd.grad(objective, z, create_graph=True)[0]
    signs = experiment.sign_vector()
    field = signs * gradient
    hessian_times_field = torch.autograd.grad(
        gradient, z, grad_outputs=field.detach()
    )[0]
    return field.detach(), (signs * hessian_times_field).detach()


def cosine(left, right):
    return float(
        torch.dot(left, right)
        / (torch.linalg.norm(left) * torch.linalg.norm(right) + 1.0e-15)
    )


def main():
    game = experiment.population.game_catalog()["CyclicControl"]
    z = experiment.initialize(3100, game)
    batch_a = experiment.sample_batch(z, game, 2048, seed=271828)
    batch_b = experiment.sample_batch(z, game, 2048, seed=271828)
    assert torch.equal(batch_a.states, batch_b.states)
    assert torch.equal(batch_a.protagonist_actions, batch_b.protagonist_actions)
    assert torch.equal(batch_a.adversary_actions, batch_b.adversary_actions)
    assert torch.equal(batch_a.rewards, batch_b.rewards)
    assert torch.equal(batch_a.behavior_log_joint, batch_b.behavior_log_joint)

    z_probe = z.detach().clone().requires_grad_(True)
    _, ratios = experiment.stochastic_objective(z_probe, batch_a, game)
    behavior_ratio_error = float(torch.max(torch.abs(ratios.detach() - 1.0)))
    assert behavior_ratio_error <= 1.0e-12

    field, curvature = experiment.stochastic_oracles(z, batch_a, game)
    assert field.shape == (experiment.TOTAL_DIM,)
    assert curvature.shape == (experiment.TOTAL_DIM,)
    assert torch.isfinite(field).all()
    assert torch.isfinite(curvature).all()

    radius = min(
        1.0e-4,
        1.0e-4 / max(float(torch.linalg.norm(field)), 1.0e-12),
    )
    plus = experiment.empirical_field(z + radius * field, batch_a, game)
    minus = experiment.empirical_field(z - radius * field, batch_a, game)
    finite_difference = (plus - minus) / (2.0 * radius)
    relative_hvp_error = float(
        torch.linalg.norm(curvature - finite_difference)
        / (torch.linalg.norm(curvature) + 1.0e-15)
    )
    assert relative_hvp_error <= 1.0e-4

    exact_field, exact_curvature = finite_horizon_population_oracles(z, game)
    sampled_fields = []
    sampled_curvatures = []
    for index in range(64):
        batch = experiment.sample_batch(
            z, game, 2048, seed=5_000_000 + index
        )
        sampled_field, sampled_curvature = experiment.stochastic_oracles(
            z, batch, game
        )
        sampled_fields.append(sampled_field)
        sampled_curvatures.append(sampled_curvature)
    mean_field = torch.stack(sampled_fields).mean(dim=0)
    mean_curvature = torch.stack(sampled_curvatures).mean(dim=0)
    field_cosine = cosine(mean_field, exact_field)
    curvature_cosine = cosine(mean_curvature, exact_curvature)
    assert field_cosine >= 0.90
    assert curvature_cosine >= 0.80

    scale = experiment.normalizer(z, batch_a, game)
    coefficients = experiment.directional_coefficients(
        z, field, curvature, batch_a, game, scale, True
    )
    assert coefficients["c_hat"] > 0.0
    assert coefficients["a_hat"] > 0.0
    assert coefficients["determinant"] > 0.0
    beta, gamma, predicted = experiment.population.solve_box(
        coefficients, True
    )
    candidate, beta_used, gamma_used, _, realized = (
        experiment.safeguarded_step(
            z,
            field,
            curvature,
            beta,
            gamma,
            batch_a,
            game,
            scale,
        )
    )
    assert torch.isfinite(candidate).all()
    assert 0.0 <= beta_used <= experiment.BETA_MAX
    assert 0.0 <= gamma_used <= experiment.GAMMA_MAX
    assert realized >= -1.0e-10

    metrics = experiment.exact_metrics(z, game)
    assert metrics["hard_br_residual"] <= 1.0e-10
    assert metrics["soft_br_residual"] <= 1.0e-8
    report = {
        "same_seed_batch_reproducible": True,
        "field_dimension": experiment.TOTAL_DIM,
        "behavior_ratio_max_error": behavior_ratio_error,
        "same_batch_autograd_vs_fd_hvp_relative_error": relative_hvp_error,
        "mean_oracle_batches": 64,
        "mean_oracle_transitions_per_batch": 2048,
        "mean_field_to_finite_horizon_population_cosine": field_cosine,
        "mean_curvature_to_finite_horizon_population_cosine": curvature_cosine,
        "qp_beta": beta,
        "qp_gamma": gamma,
        "qp_predicted_decrease": -predicted,
        "accepted_beta": beta_used,
        "accepted_gamma": gamma_used,
        "accepted_same_batch_merit_decrease": realized,
        "hard_br_residual": metrics["hard_br_residual"],
        "soft_br_residual": metrics["soft_br_residual"],
    }
    output = Path(__file__).resolve().parent / "results" / (
        "sentinels-dice-" + time.strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"OUTPUT={output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
