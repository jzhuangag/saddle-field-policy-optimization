"""Numerical sentinels for the manuscript's exact algebraic identities.

These tests do not replace proofs.  They detect sign, transpose, active-set,
and entropy-constant mistakes in the formulas implemented by the paper suite.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp


def box_candidates(e, d, c, a, b, beta_max, gamma_max):
    determinant = c * a - b * b
    q = lambda x: -e * x[0] - d * x[1] + 0.5 * c * x[0] ** 2 - b * x[0] * x[1] + 0.5 * a * x[1] ** 2
    points = [(0.0, 0.0)]
    interior = ((a * e + b * d) / determinant, (b * e + c * d) / determinant)
    if 0 <= interior[0] <= beta_max and 0 <= interior[1] <= gamma_max:
        points.append(interior)
    points.extend([(np.clip(e / c, 0, beta_max), 0.0), (0.0, np.clip(d / a, 0, gamma_max)), (beta_max, np.clip((d + b * beta_max) / a, 0, gamma_max)), (np.clip((e + b * gamma_max) / c, 0, beta_max), gamma_max)])
    return min(points, key=q), min(map(q, points))


def matrix_game_value(matrix):
    rows, cols = matrix.shape
    # maximize v subject to M^T p >= v 1, p>=0, 1^T p=1
    objective = np.r_[np.zeros(rows), -1.0]
    aub = np.c_[-matrix.T, np.ones(cols)]
    result = linprog(objective, A_ub=aub, b_ub=np.zeros(cols), A_eq=np.r_[np.ones(rows), 0.0][None, :], b_eq=[1.0], bounds=[(0, None)] * rows + [(None, None)], method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[-1]


def entropy(probability):
    return -float(np.sum(probability * np.log(np.maximum(probability, 1e-300))))


def main():
    rng = np.random.RandomState(20260723)
    max_decomposition_error = 0.0
    for _ in range(1000):
        dimension = rng.randint(2, 9)
        jacobian = rng.normal(size=(dimension, dimension))
        f = rng.normal(size=dimension)
        s = 0.5 * (jacobian + jacobian.T); w = 0.5 * (jacobian - jacobian.T)
        grad = jacobian.T @ f; g = jacobian @ f
        errors = [abs(grad @ (-f) + f @ s @ f), abs(grad @ g - (np.linalg.norm(s @ f) ** 2 - np.linalg.norm(w @ f) ** 2))]
        max_decomposition_error = max(max_decomposition_error, *errors)

    max_dominance_error = 0.0
    max_box_objective_error = 0.0
    for _ in range(1000):
        factor = rng.normal(size=(2, 2))
        hessian = factor.T @ factor + 0.2 * np.eye(2)
        c, a, b = hessian[0, 0], hessian[1, 1], -hessian[0, 1]
        target = rng.uniform(0.0, 0.2, size=2)
        e, d = hessian @ target
        determinant = c * a - b * b
        delta_star = 0.5 * np.array([e, d]) @ np.linalg.solve(hessian, np.array([e, d]))
        delta_f = max(e, 0.0) ** 2 / (2 * c)
        formula = (b * e + c * d) ** 2 / (2 * c * determinant) + max(-e, 0.0) ** 2 / (2 * c)
        max_dominance_error = max(max_dominance_error, abs((delta_star - delta_f) - formula))
        beta_max, gamma_max = rng.uniform(0.01, 0.2, size=2)
        candidate, objective = box_candidates(e, d, c, a, b, beta_max, gamma_max)
        q = lambda x: -e * x[0] - d * x[1] + 0.5 * c * x[0] ** 2 - b * x[0] * x[1] + 0.5 * a * x[1] ** 2
        numerical = minimize(q, x0=np.array([beta_max / 2, gamma_max / 2]), bounds=[(0, beta_max), (0, gamma_max)], method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000})
        max_box_objective_error = max(max_box_objective_error, abs(objective - numerical.fun))

    alpha, epsilon = 0.9, 0.03
    max_bridge_violation = 0.0
    for _ in range(500):
        payoff = rng.normal(size=(3, 3))
        p = rng.dirichlet(np.ones(3)); q = rng.dirichlet(np.ones(3))
        br_return = float(np.min(p @ payoff)) / (1 - alpha)
        value = matrix_game_value(payoff) / (1 - alpha)
        gap0 = (float(np.max(payoff @ q)) - float(np.min(p @ payoff))) / (1 - alpha)
        soft_max = (epsilon * logsumexp((payoff @ q) / epsilon) - epsilon * entropy(q)) / (1 - alpha)
        soft_min = (-epsilon * logsumexp(-(p @ payoff) / epsilon) + epsilon * entropy(p)) / (1 - alpha)
        gap_epsilon = soft_max - soft_min
        bias = epsilon * (math.log(3) + math.log(3)) / (1 - alpha)
        violations = [-(value - br_return), (value - br_return) - gap0, gap0 - gap_epsilon - bias]
        max_bridge_violation = max(max_bridge_violation, *violations)

    max_rotation_margin_error = 0.0
    for radius in np.geomspace(1.0e-8, 1.0e2, 200):
        phi = radius**2
        c = d = radius**2
        determinant = radius**4
        dominance_gap = (c * d) ** 2 / (2.0 * c * determinant)
        max_rotation_margin_error = max(
            max_rotation_margin_error, abs(dominance_gap / phi - 0.5)
        )

    max_centered_softplus_violation = 0.0
    for u in np.geomspace(1.0e-12, 1.0e2, 500):
        centered = epsilon * np.logaddexp(0.0, u / epsilon) - epsilon * math.log(2.0)
        violations = [-centered, centered - u, (u - centered) - epsilon * math.log(2.0)]
        max_centered_softplus_violation = max(max_centered_softplus_violation, *violations)

    status = (
        max_decomposition_error < 1e-10
        and max_dominance_error < 1e-10
        and max_box_objective_error < 1e-8
        and max_bridge_violation < 1e-10
        and max_rotation_margin_error < 1e-12
        and max_centered_softplus_violation < 1e-12
    )
    report = {
        "trials": {
            "decomposition": 1000,
            "qp": 1000,
            "entropy_bridge": 500,
            "pure_rotation_radii": 200,
            "centered_softplus": 500,
        },
        "maximum_decomposition_identity_error": max_decomposition_error,
        "maximum_corrected_dominance_identity_error": max_dominance_error,
        "maximum_box_qp_objective_error_vs_scipy": max_box_objective_error,
        "maximum_performance_bridge_violation": max(0.0, max_bridge_violation),
        "maximum_pure_rotation_normalized_margin_error_from_one_half": max_rotation_margin_error,
        "maximum_centered_softplus_property_violation": max(0.0, max_centered_softplus_violation),
        "status": "PASS" if status else "FAIL",
    }
    output = Path(__file__).resolve().parent / "results" / ("theory-sentinels-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESULT_DIR=" + str(output)); print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
