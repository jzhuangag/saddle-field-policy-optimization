"""Exact controlled geometry experiment for Section VI-A.

The field F(z)=(mu I + sigma J)z is the local normal form of a two-dimensional
zero-sum saddle field.  For V=||F||^2/2 and G=DF F, the QP model is exact.
This script verifies the analytical skew threshold sigma/mu=1 and compares all
six methods without stochastic or finite-difference error.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LR = 0.03
BETA_MAX = 0.03
GAMMA_MAX = 0.03
STEPS = 120
METHODS = ("QP+G", "noG", "GDA", "Adam-GDA", "EGM", "PPM-3")
RATIOS = np.linspace(0.0, 2.5, 51)


def matrix(mu: float, sigma: float) -> np.ndarray:
    return np.array([[mu, sigma], [-sigma, mu]], dtype=np.float64)


def field(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return a @ z


def merit(z: np.ndarray, a: np.ndarray) -> float:
    f = field(z, a)
    return 0.5 * float(f @ f)


def coefficients(z: np.ndarray, a: np.ndarray) -> dict[str, float]:
    f = a @ z
    g = a @ f
    h = a.T @ a
    grad = h @ z
    e = float(grad @ f)
    d = -float(grad @ g)
    c = float(f @ h @ f)
    aa = float(g @ h @ g)
    b = float(f @ h @ g)
    return {"e": e, "d": d, "c": c, "a": aa, "b": b, "delta": c * aa - b * b}


def q_value(beta: float, gamma: float, c: dict[str, float]) -> float:
    return -beta * c["e"] - gamma * c["d"] + 0.5 * c["c"] * beta**2 - c["b"] * beta * gamma + 0.5 * c["a"] * gamma**2


def solve_box(c: dict[str, float], use_g: bool) -> tuple[float, float, float]:
    eps = 1.0e-14
    if not use_g:
        beta = float(np.clip(c["e"] / max(c["c"], eps), 0.0, BETA_MAX))
        return beta, 0.0, q_value(beta, 0.0, c)
    candidates = [(0.0, 0.0)]
    if c["delta"] > eps:
        beta = (c["a"] * c["e"] + c["b"] * c["d"]) / c["delta"]
        gamma = (c["b"] * c["e"] + c["c"] * c["d"]) / c["delta"]
        if 0.0 <= beta <= BETA_MAX and 0.0 <= gamma <= GAMMA_MAX:
            candidates.append((beta, gamma))
    candidates.extend(
        [
            (float(np.clip(c["e"] / max(c["c"], eps), 0.0, BETA_MAX)), 0.0),
            (0.0, float(np.clip(c["d"] / max(c["a"], eps), 0.0, GAMMA_MAX))),
            (BETA_MAX, float(np.clip((c["d"] + c["b"] * BETA_MAX) / max(c["a"], eps), 0.0, GAMMA_MAX))),
            (float(np.clip((c["e"] + c["b"] * GAMMA_MAX) / max(c["c"], eps), 0.0, BETA_MAX)), GAMMA_MAX),
            (BETA_MAX, 0.0),
            (0.0, GAMMA_MAX),
            (BETA_MAX, GAMMA_MAX),
        ]
    )
    beta, gamma = min(candidates, key=lambda pair: q_value(pair[0], pair[1], c))
    return beta, gamma, q_value(beta, gamma, c)


def classical_step(method: str, z: np.ndarray, a: np.ndarray, adam):
    f = field(z, a)
    if method == "GDA":
        return z - LR * f, adam
    if method == "EGM":
        return z - LR * field(z - LR * f, a), adam
    if method == "PPM-3":
        iterate = z.copy()
        for _ in range(3):
            iterate = z - LR * field(iterate, a)
        return iterate, adam
    if adam is None:
        first, second, count = np.zeros_like(z), np.zeros_like(z), 0
    else:
        first, second, count = adam
    count += 1
    first = 0.9 * first + 0.1 * f
    second = 0.999 * second + 0.001 * np.square(f)
    direction = (first / (1.0 - 0.9**count)) / (np.sqrt(second / (1.0 - 0.999**count)) + 1.0e-8)
    return z - LR * direction, (first, second, count)


def run_method(method: str, mu: float, sigma: float, steps: int = STEPS):
    a = matrix(mu, sigma)
    z = np.array([1.0, -0.65], dtype=np.float64)
    adam = None
    rows = []
    for step in range(steps + 1):
        f = field(z, a)
        g = a @ f
        c = coefficients(z, a)
        grad = a.T @ f
        s = 0.5 * (a + a.T)
        w = 0.5 * (a - a.T)
        rows.append(
            {
                "method": method,
                "mu": mu,
                "sigma": sigma,
                "ratio": sigma / mu if mu > 0 else math.inf,
                "step": step,
                "z0": z[0],
                "z1": z[1],
                "merit": merit(z, a),
                "field_norm": float(np.linalg.norm(f)),
                "d": c["d"],
                "analytic_d": (mu * mu + sigma * sigma) * (sigma * sigma - mu * mu) * float(z @ z),
                "skew_ratio": float(np.linalg.norm(w @ f) / max(np.linalg.norm(s @ f), 1.0e-15)),
                "cos_fg": float(f @ g / max(np.linalg.norm(f) * np.linalg.norm(g), 1.0e-15)),
                "grad_norm": float(np.linalg.norm(grad)),
            }
        )
        if step == steps:
            break
        if method in ("QP+G", "noG"):
            beta, gamma, predicted = solve_box(c, method == "QP+G")
            candidate = z - beta * f + gamma * g
            realized = merit(z, a) - merit(candidate, a)
            rows[-1].update({"beta": beta, "gamma": gamma, "predicted_decrease": -predicted, "realized_decrease": realized})
            z = candidate
        else:
            z, adam = classical_step(method, z, a, adam)
    return rows


def main() -> None:
    started = time.time()
    rows = []
    mu = 0.35
    for ratio in RATIOS:
        sigma = mu * ratio
        for method in METHODS:
            rows.extend(run_method(method, mu, sigma))
    rotation_rows = []
    for method in METHODS:
        rotation_rows.extend(run_method(method, 0.0, 3.0))
    rows.extend(rotation_rows)

    root = Path(__file__).resolve().parent
    output = root / "results" / ("linear-geometry-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        names = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)

    styles = {
        "QP+G": ("black", "-"), "noG": ("tab:green", "-"),
        "GDA": ("tab:orange", "--"), "Adam-GDA": ("tab:blue", "--"),
        "EGM": ("tab:purple", "-."), "PPM-3": ("tab:red", ":"),
    }
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.25))
    ratio_data = [r for r in rows if math.isfinite(r["ratio"]) and r["step"] == STEPS]
    for method in METHODS:
        selected = [r for r in ratio_data if r["method"] == method]
        axes[0].semilogy([r["ratio"] for r in selected], [max(r["merit"], 1e-30) for r in selected], label=method, color=styles[method][0], linestyle=styles[method][1])
    axes[0].axvline(1.0, color="0.5", linewidth=1.0, linestyle=":")
    axes[0].set(xlabel=r"rotation ratio $\sigma/\mu$", ylabel=r"final $V/V_0$ (log scale)", title="phase diagram")
    # Normalize final merit by the common initial value for each ratio.
    for line, method in zip(axes[0].lines[:len(METHODS)], METHODS):
        selected = [r for r in ratio_data if r["method"] == method]
        normalized = []
        for r in selected:
            initial = next(x["merit"] for x in rows if x["method"] == method and x["ratio"] == r["ratio"] and x["step"] == 0)
            normalized.append(max(r["merit"] / initial, 1e-30))
        line.set_ydata(normalized)

    analytic = []
    numeric = []
    for ratio in RATIOS:
        sigma = mu * float(ratio)
        z0 = np.array([1.0, -0.65], dtype=np.float64)
        c0 = coefficients(z0, matrix(mu, sigma))
        scale = (mu * mu + (mu * ratio) ** 2) ** 2 * (1.0 + 0.65**2)
        analytic_d = (mu * mu + sigma * sigma) * (sigma * sigma - mu * mu) * float(z0 @ z0)
        analytic.append(analytic_d / max(scale, 1e-15))
        numeric.append(c0["d"] / max(scale, 1e-15))
    axes[1].plot(RATIOS, analytic, color="black", label="analytic")
    axes[1].plot(RATIOS, numeric, color="tab:red", linestyle="--", label="computed")
    axes[1].axhline(0.0, color="0.5", linewidth=1.0); axes[1].axvline(1.0, color="0.5", linewidth=1.0, linestyle=":")
    axes[1].set(xlabel=r"rotation ratio $\sigma/\mu$", ylabel="normalized curvature descent $d$", title="exact skew threshold")

    for method in METHODS:
        selected = [r for r in rotation_rows if r["method"] == method]
        initial = selected[0]["merit"]
        axes[2].semilogy([r["step"] for r in selected], [max(r["merit"] / initial, 1e-30) for r in selected], label=method, color=styles[method][0], linestyle=styles[method][1])
    axes[2].set(xlabel="joint update", ylabel=r"$V_k/V_0$", title="pure rotation")

    selected = [r for r in rotation_rows if r["method"] == "QP+G" and r["step"] < STEPS]
    axes[3].scatter([r["predicted_decrease"] for r in selected], [r["realized_decrease"] for r in selected], s=16, color="black")
    limit = max([r["predicted_decrease"] for r in selected] + [1e-12])
    axes[3].plot([0, limit], [0, limit], color="tab:red", linestyle="--", linewidth=1.0)
    axes[3].set(xlabel="QP predicted decrease", ylabel="realized decrease", title="quadratic-model identity")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output / "linear_geometry.pdf", bbox_inches="tight")
    fig.savefig(output / "linear_geometry.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    max_d_error = max(abs(r["d"] - r["analytic_d"]) for r in rows)
    model_error = max(abs(r.get("predicted_decrease", 0.0) - r.get("realized_decrease", 0.0)) for r in rows if "predicted_decrease" in r)
    summary = {
        "field": "F(z)=(mu I + sigma J)z",
        "merit": "V=0.5||F||^2",
        "direction": "G=DF F",
        "methods": METHODS,
        "fixed_lr": LR,
        "qp_caps": [BETA_MAX, GAMMA_MAX],
        "steps": STEPS,
        "analytic_identity": "d=(mu^2+sigma^2)(sigma^2-mu^2)||z||^2",
        "maximum_d_identity_error": max_d_error,
        "maximum_quadratic_model_error": model_error,
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RESULT_DIR=" + str(output), flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
