"""Historical sentinels for the rejected fixed-trajectory prototype.

The paper-facing sentinels are in ``sentinels_dice.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import stochastic_actor_critic as experiment


def central_curvature(z, field, batch, game, radius):
    plus, _ = experiment.empirical_field(z + radius * field, batch, game)
    minus, _ = experiment.empirical_field(z - radius * field, batch, game)
    return (plus - minus) / (2.0 * radius)


def main():
    game = experiment.population.game_catalog()["CyclicControl"]
    z = experiment.initialize(8128, game)
    batch_a = experiment.sample_batch(z, game, 128, seed=271828)
    batch_b = experiment.sample_batch(z, game, 128, seed=271828)
    assert torch.equal(batch_a.states, batch_b.states)
    assert torch.equal(batch_a.protagonist_actions, batch_b.protagonist_actions)
    assert torch.equal(batch_a.adversary_actions, batch_b.adversary_actions)
    assert torch.equal(batch_a.rewards, batch_b.rewards)

    field, critic_loss = experiment.empirical_field(z, batch_a, game)
    assert field.shape == (experiment.TOTAL_DIM,)
    assert torch.isfinite(field).all()
    assert np.isfinite(critic_loss)

    base_radius = min(
        1.0e-2,
        experiment.CURVATURE_RADIUS
        / max(float(torch.linalg.norm(field)), 1.0e-12),
    )
    curvature_h = central_curvature(z, field, batch_a, game, base_radius)
    curvature_half = central_curvature(
        z, field, batch_a, game, base_radius / 2.0
    )
    relative_curvature_change = float(
        torch.linalg.norm(curvature_h - curvature_half)
        / (torch.linalg.norm(curvature_half) + 1.0e-15)
    )
    assert torch.isfinite(curvature_h).all()
    assert relative_curvature_change < 0.10

    scales = experiment.normalizers(z, batch_a, game)
    coefficients = experiment.directional_coefficients(
        z, field, curvature_half, batch_a, game, scales, True
    )
    assert coefficients["c_hat"] > 0.0
    assert coefficients["a_hat"] > 0.0
    assert coefficients["determinant"] > 0.0
    beta, gamma, predicted = experiment.population.solve_box(coefficients, True)
    assert 0.0 <= beta <= experiment.BETA_MAX
    assert 0.0 <= gamma <= experiment.GAMMA_MAX
    candidate, beta_used, gamma_used, _, realized = experiment.safeguarded_step(
        z,
        field,
        curvature_half,
        beta,
        gamma,
        batch_a,
        game,
        scales,
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
        "field_norm": float(torch.linalg.norm(field)),
        "critic_loss": critic_loss,
        "curvature_radius": base_radius,
        "relative_curvature_change_h_to_half_h": relative_curvature_change,
        "qp_beta": beta,
        "qp_gamma": gamma,
        "qp_predicted_decrease": -predicted,
        "accepted_beta": beta_used,
        "accepted_gamma": gamma_used,
        "accepted_empirical_merit_decrease": realized,
        "hard_br_residual": metrics["hard_br_residual"],
        "soft_br_residual": metrics["soft_br_residual"],
    }
    output = Path(__file__).resolve().parent / "results" / (
        "sentinels-" + time.strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"OUTPUT={output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
