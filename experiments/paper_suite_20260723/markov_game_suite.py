"""Final exact-gap tabular/neural zero-sum Markov-game experiments.

The regularized policy-space Nash gap is recomputed at every directional
stencil point.  Hard unregularized best responses are used only for evaluation.
All policies are updated simultaneously and there is no warm-up.
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
from scipy import optimize, stats
from scipy.special import logsumexp

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

from journal_games import (
    Game,
    journal_games,
    policy as neural_policy,
    policy_dim as neural_policy_dim,
)

DISCOUNT = 0.90
ENTROPY_TAU = 0.03
LR = 0.03
BETA_MAX = 0.03
GAMMA_MAX = 0.03
PROBE_RADIUS = 1.0e-3
PD_FLOOR = 1.0e-8
BACKTRACK_MAX = 12
LAMBDA_F = 0.3
LAMBDA_GAP = 1.0
METHODS = ("QP+G", "noG", "GDA", "Adam-GDA", "EGM", "PPM-3")
CHECKPOINT_EVERY = 5
GAME_VALUE_CACHE: dict[str, tuple[float, float, int]] = {}


def rps_game() -> Game:
    reward = torch.tensor([[[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]])
    transition = torch.ones((1, 3, 3, 1))
    features = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    return Game("RPS", features, reward, transition, torch.ones(1))


def game_catalog() -> dict[str, Game]:
    games = [rps_game(), *journal_games()]
    return {game.name: game for game in games}


def parameter_dim(mode: str, game: Game) -> int:
    block = len(game.rho) * 3 if mode == "tabular" else neural_policy_dim()
    return 2 * block


def split_policies(z: torch.Tensor, game: Game, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "tabular":
        block = len(game.rho) * 3
        return torch.softmax(z[:block].reshape(len(game.rho), 3), dim=1), torch.softmax(z[block:].reshape(len(game.rho), 3), dim=1)
    block = neural_policy_dim()
    return neural_policy(z[:block], game.features), neural_policy(z[block:], game.features)


def entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(probability * torch.log(probability.clamp_min(1.0e-15))).sum(dim=1)


def return_from_policies(p: torch.Tensor, q: torch.Tensor, game: Game, regularized: bool) -> torch.Tensor:
    reward = torch.einsum("sa,sab,sb->s", p, game.rewards, q)
    transition = torch.einsum("sa,sabn,sb->sn", p, game.transitions, q)
    if regularized:
        reward = reward + ENTROPY_TAU * entropy(p) - ENTROPY_TAU * entropy(q)
    value = torch.linalg.solve(torch.eye(len(game.rho)) - DISCOUNT * transition, reward)
    return game.rho @ value


def return_value(z: torch.Tensor, game: Game, mode: str, regularized: bool) -> torch.Tensor:
    return return_from_policies(*split_policies(z, game, mode), game, regularized)


def sign_vector(mode: str, game: Game) -> torch.Tensor:
    block = parameter_dim(mode, game) // 2
    return torch.cat((-torch.ones(block), torch.ones(block)))


def field(z_value: torch.Tensor, game: Game, mode: str, create_graph: bool = False) -> torch.Tensor:
    z = z_value if z_value.requires_grad else z_value.detach().clone().requires_grad_(True)
    objective = return_value(z, game, mode, regularized=True)
    gradient = torch.autograd.grad(objective, z, create_graph=create_graph)[0]
    return sign_vector(mode, game) * gradient


def field_and_geometry(z_value: torch.Tensor, game: Game, mode: str):
    z = z_value.detach().clone().requires_grad_(True)
    objective = return_value(z, game, mode, regularized=True)
    gradient = torch.autograd.grad(objective, z, create_graph=True)[0]
    signs = sign_vector(mode, game)
    f = signs * gradient
    f_detached = f.detach()
    h_f = torch.autograd.grad(gradient, z, grad_outputs=f_detached, retain_graph=True)[0]
    g = signs * h_f
    at_f = torch.autograd.grad(gradient, z, grad_outputs=signs * f_detached)[0]
    symmetric_action = 0.5 * (g + at_f)
    skew_action = 0.5 * (g - at_f)
    rotation = float(torch.linalg.norm(skew_action) / (torch.linalg.norm(symmetric_action) + 1.0e-15))
    cosine = float(torch.dot(f_detached, g.detach()) / (torch.linalg.norm(f_detached) * torch.linalg.norm(g.detach()) + 1.0e-15))
    return f_detached, g.detach(), rotation, cosine


def soft_br_bank(z: torch.Tensor, game: Game, mode: str, tolerance: float = 1.0e-10, max_iterations: int = 500):
    with torch.no_grad():
        p_t, q_t = split_policies(z, game, mode)
        p, q = p_t.cpu().numpy(), q_t.cpu().numpy()
        rewards, transitions = game.rewards.cpu().numpy(), game.transitions.cpu().numpy()
        hp = -(p * np.log(np.maximum(p, 1.0e-15))).sum(axis=1)
        hq = -(q * np.log(np.maximum(q, 1.0e-15))).sum(axis=1)
        rp = np.einsum("sab,sb->sa", rewards, q) - ENTROPY_TAU * hq[:, None]
        pp = np.einsum("sabn,sb->san", transitions, q)
        ra = np.einsum("sa,sab->sb", p, rewards) + ENTROPY_TAU * hp[:, None]
        pa = np.einsum("sa,sabn->sbn", p, transitions)
        vp = np.zeros(len(game.rho)); va = np.zeros(len(game.rho))
        for iteration in range(1, max_iterations + 1):
            qp = rp + DISCOUNT * np.einsum("san,n->sa", pp, vp)
            qa = ra + DISCOUNT * np.einsum("sbn,n->sb", pa, va)
            new_vp = ENTROPY_TAU * logsumexp(qp / ENTROPY_TAU, axis=1)
            new_va = -ENTROPY_TAU * logsumexp(-qa / ENTROPY_TAU, axis=1)
            if max(float(np.max(np.abs(new_vp - vp))), float(np.max(np.abs(new_va - va)))) <= tolerance:
                vp, va = new_vp, new_va
                break
            vp, va = new_vp, new_va
        qp = rp + DISCOUNT * np.einsum("san,n->sa", pp, vp)
        qa = ra + DISCOUNT * np.einsum("sbn,n->sb", pa, va)
        tp = ENTROPY_TAU * logsumexp(qp / ENTROPY_TAU, axis=1)
        ta = -ENTROPY_TAU * logsumexp(-qa / ENTROPY_TAU, axis=1)
        residual = max(float(np.max(np.abs(tp - vp))), float(np.max(np.abs(ta - va))))
        p_br = np.exp(qp / ENTROPY_TAU - logsumexp(qp / ENTROPY_TAU, axis=1, keepdims=True))
        q_br = np.exp(-qa / ENTROPY_TAU - logsumexp(-qa / ENTROPY_TAU, axis=1, keepdims=True))
        return torch.tensor(p_br), torch.tensor(q_br), residual, iteration


def regularized_gap(z: torch.Tensor, game: Game, mode: str) -> tuple[float, float, int]:
    protagonist_br, adversary_br, residual, iterations = soft_br_bank(z, game, mode)
    p, q = split_policies(z, game, mode)
    br_max = float(return_from_policies(protagonist_br, q, game, regularized=True).detach())
    br_min = float(return_from_policies(p, adversary_br, game, regularized=True).detach())
    return max(0.0, br_max - br_min), residual, iterations


def hard_br_values(z: torch.Tensor, game: Game, mode: str, tolerance: float = 1.0e-12, max_iterations: int = 800):
    with torch.no_grad():
        p_t, q_t = split_policies(z, game, mode)
        p, q = p_t.cpu().numpy(), q_t.cpu().numpy()
        rewards, transitions = game.rewards.cpu().numpy(), game.transitions.cpu().numpy()
        rp = np.einsum("sab,sb->sa", rewards, q)
        pp = np.einsum("sabn,sb->san", transitions, q)
        ra = np.einsum("sa,sab->sb", p, rewards)
        pa = np.einsum("sa,sabn->sbn", p, transitions)
        vp = np.zeros(len(game.rho)); va = np.zeros(len(game.rho))
        for iteration in range(1, max_iterations + 1):
            new_vp = (rp + DISCOUNT * np.einsum("san,n->sa", pp, vp)).max(axis=1)
            new_va = (ra + DISCOUNT * np.einsum("sbn,n->sb", pa, va)).min(axis=1)
            if max(float(np.max(np.abs(new_vp - vp))), float(np.max(np.abs(new_va - va)))) <= tolerance:
                vp, va = new_vp, new_va
                break
            vp, va = new_vp, new_va
        tp = (rp + DISCOUNT * np.einsum("san,n->sa", pp, vp)).max(axis=1)
        ta = (ra + DISCOUNT * np.einsum("sbn,n->sb", pa, va)).min(axis=1)
        residual = max(float(np.max(np.abs(tp - vp))), float(np.max(np.abs(ta - va))))
        rho = game.rho.cpu().numpy()
        return float(rho @ vp), float(rho @ va), residual, iteration


def matrix_game_value(payoff: np.ndarray) -> float:
    """Return max_p min_b p^T payoff[:, b] for a finite zero-sum matrix game."""
    actions_p, actions_a = payoff.shape
    objective = np.r_[np.zeros(actions_p), -1.0]
    inequalities = np.zeros((actions_a, actions_p + 1))
    inequalities[:, :actions_p] = -payoff.T
    inequalities[:, -1] = 1.0
    result = optimize.linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(actions_a),
        A_eq=np.r_[np.ones(actions_p), 0.0][None, :],
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * actions_p + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError("matrix-game LP failed: " + result.message)
    return float(result.x[-1])


def hard_game_value(game: Game, tolerance: float = 1.0e-12, max_iterations: int = 1000):
    """Compute the unregularized discounted Shapley value."""
    if game.name in GAME_VALUE_CACHE:
        return GAME_VALUE_CACHE[game.name]
    rewards = game.rewards.cpu().numpy()
    transitions = game.transitions.cpu().numpy()
    value = np.zeros(len(game.rho))
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        updated = np.empty_like(value)
        for state in range(len(value)):
            continuation = np.einsum("abn,n->ab", transitions[state], value)
            updated[state] = matrix_game_value(rewards[state] + DISCOUNT * continuation)
        residual = float(np.max(np.abs(updated - value)))
        value = updated
        if residual <= tolerance:
            bellman = np.empty_like(value)
            for state in range(len(value)):
                continuation = np.einsum("abn,n->ab", transitions[state], value)
                bellman[state] = matrix_game_value(
                    rewards[state] + DISCOUNT * continuation
                )
            bellman_residual = float(np.max(np.abs(bellman - value)))
            answer = (
                float(game.rho.cpu().numpy() @ value),
                bellman_residual,
                iteration,
            )
            GAME_VALUE_CACHE[game.name] = answer
            return answer
    raise RuntimeError(f"hard game-value iteration failed for {game.name}: residual={residual}")


def raw_components(z: torch.Tensor, game: Game, mode: str):
    f = field(z, game, mode)
    gap, residual, iterations = regularized_gap(z, game, mode)
    return 0.5 * float(f @ f), gap, residual, iterations


def merit(z: torch.Tensor, game: Game, mode: str, normalizers: tuple[float, float]):
    energy, gap, residual, iterations = raw_components(z, game, mode)
    return LAMBDA_F * energy / normalizers[0] + LAMBDA_GAP * gap / normalizers[1], residual, iterations


def directional_coefficients(z: torch.Tensor, f: torch.Tensor, g: torch.Tensor, game: Game, mode: str, normalizers: tuple[float, float], use_g: bool):
    hf = min(1.0e-2, PROBE_RADIUS / max(float(torch.linalg.norm(f)), 1.0e-12))
    points = {"v0": z, "vfp": z + hf * f, "vfm": z - hf * f}
    if use_g:
        hg = min(1.0e-2, PROBE_RADIUS / max(float(torch.linalg.norm(g)), 1.0e-12))
        points.update({"vgp": z + hg * g, "vgm": z - hg * g, "vpp": z + hf * f + hg * g, "vpm": z + hf * f - hg * g, "vmp": z - hf * f + hg * g, "vmm": z - hf * f - hg * g})
    values = {}; residuals = []; iterations = []
    for key, point in points.items():
        values[key], residual, count = merit(point, game, mode, normalizers)
        residuals.append(residual); iterations.append(count)
    e = (values["vfp"] - values["vfm"]) / (2.0 * hf)
    c_raw = (values["vfp"] - 2.0 * values["v0"] + values["vfm"]) / hf**2
    if not use_g:
        c_hat = c_raw + max(0.0, PD_FLOOR - c_raw)
        return {"e": e, "d": 0.0, "c_hat": c_hat, "a_hat": 1.0, "b": 0.0, "determinant": c_hat, "inflation": max(0.0, PD_FLOOR - c_raw), "max_soft_residual": max(residuals), "max_soft_iterations": max(iterations)}
    d = -(values["vgp"] - values["vgm"]) / (2.0 * hg)
    a_raw = (values["vgp"] - 2.0 * values["v0"] + values["vgm"]) / hg**2
    b = (values["vpp"] - values["vpm"] - values["vmp"] + values["vmm"]) / (4.0 * hf * hg)
    hessian = np.array([[c_raw, -b], [-b, a_raw]])
    inflation = max(0.0, PD_FLOOR - float(np.linalg.eigvalsh(hessian)[0]))
    c_hat, a_hat = c_raw + inflation, a_raw + inflation
    determinant = c_hat * a_hat - b * b
    if determinant <= 1.0e-14:
        extra = math.sqrt(abs(determinant)) + PD_FLOOR
        inflation += extra; c_hat += extra; a_hat += extra
        determinant = c_hat * a_hat - b * b
    return {"e": e, "d": d, "c_hat": c_hat, "a_hat": a_hat, "b": b, "determinant": determinant, "inflation": inflation, "max_soft_residual": max(residuals), "max_soft_iterations": max(iterations)}


def q_value(beta: float, gamma: float, c) -> float:
    return -beta * c["e"] - gamma * c["d"] + 0.5 * c["c_hat"] * beta**2 - c["b"] * beta * gamma + 0.5 * c["a_hat"] * gamma**2


def solve_box(c, use_g: bool):
    if not use_g:
        beta = float(np.clip(c["e"] / c["c_hat"], 0.0, BETA_MAX))
        return beta, 0.0, q_value(beta, 0.0, c)
    candidates = [(0.0, 0.0)]
    if c["determinant"] > 1.0e-14 * max(abs(c["a_hat"] * c["c_hat"]), abs(c["b"] ** 2), 1.0):
        beta = (c["a_hat"] * c["e"] + c["b"] * c["d"]) / c["determinant"]
        gamma = (c["b"] * c["e"] + c["c_hat"] * c["d"]) / c["determinant"]
        if 0.0 <= beta <= BETA_MAX and 0.0 <= gamma <= GAMMA_MAX:
            candidates.append((beta, gamma))
    candidates.extend([(float(np.clip(c["e"] / c["c_hat"], 0.0, BETA_MAX)), 0.0), (0.0, float(np.clip(c["d"] / c["a_hat"], 0.0, GAMMA_MAX))), (BETA_MAX, float(np.clip((c["d"] + c["b"] * BETA_MAX) / c["a_hat"], 0.0, GAMMA_MAX))), (float(np.clip((c["e"] + c["b"] * GAMMA_MAX) / c["c_hat"], 0.0, BETA_MAX)), GAMMA_MAX), (BETA_MAX, 0.0), (0.0, GAMMA_MAX), (BETA_MAX, GAMMA_MAX)])
    beta, gamma = min(candidates, key=lambda pair: q_value(pair[0], pair[1], c))
    return beta, gamma, q_value(beta, gamma, c)


def safeguarded_step(z, f, g, beta, gamma, game, mode, normalizers):
    v0, r0, _ = merit(z, game, mode, normalizers)
    scale = 1.0
    maximum_residual = r0
    for backtracks in range(BACKTRACK_MAX + 1):
        bu, gu = scale * beta, scale * gamma
        candidate = z - bu * f + gu * g
        vnew, residual, _ = merit(candidate, game, mode, normalizers)
        maximum_residual = max(maximum_residual, residual)
        if vnew <= v0 + 1.0e-10:
            return candidate.detach(), vnew - v0, backtracks, bu, gu, maximum_residual
        scale *= 0.5
    return z.detach().clone(), 0.0, BACKTRACK_MAX + 1, 0.0, 0.0, maximum_residual


def classical_step(method, z, game, mode, adam_state):
    f = field(z, game, mode)
    if method == "GDA":
        return (z - LR * f).detach(), adam_state
    if method == "EGM":
        predictor = (z - LR * f).detach()
        return (z - LR * field(predictor, game, mode)).detach(), adam_state
    if method == "PPM-3":
        iterate = z.detach().clone()
        for _ in range(3):
            iterate = (z - LR * field(iterate, game, mode)).detach()
        return iterate, adam_state
    if adam_state is None:
        first, second, count = torch.zeros_like(z), torch.zeros_like(z), 0
    else:
        first, second, count = adam_state
    count += 1
    first = 0.9 * first + 0.1 * f
    second = 0.999 * second + 0.001 * torch.square(f)
    direction = (first / (1.0 - 0.9**count)) / (torch.sqrt(second / (1.0 - 0.999**count)) + 1.0e-8)
    return (z - LR * direction).detach(), (first, second, count)


def initialize(seed: int, mode: str, game: Game):
    generator = torch.Generator().manual_seed(20260723 + seed)
    return 0.08 * torch.randn(parameter_dim(mode, game), generator=generator)


def metrics(z, game, mode):
    current = float(return_value(z.detach().clone().requires_grad_(True), game, mode, regularized=False).detach())
    br_max, br_min, residual, iterations = hard_br_values(z, game, mode)
    f = field(z, game, mode)
    gap, soft_residual, soft_iterations = regularized_gap(z, game, mode)
    game_value, game_value_residual, game_value_iterations = hard_game_value(game)
    entropy_bias = ENTROPY_TAU * (
        math.log(game.rewards.shape[1]) + math.log(game.rewards.shape[2])
    ) / (1.0 - DISCOUNT)
    robust_deficiency = game_value - br_min
    bridge_upper = gap + entropy_bias
    return {
        "current_return": current,
        "hard_br_return": br_min,
        "hard_exploitability": br_max - br_min,
        "regularized_gap": gap,
        "field_norm": float(torch.linalg.norm(f)),
        "game_value": game_value,
        "robust_deficiency": robust_deficiency,
        "bridge_upper": bridge_upper,
        "bridge_slack": bridge_upper - robust_deficiency,
        "hard_br_residual": residual,
        "hard_br_iterations": iterations,
        "soft_br_residual": soft_residual,
        "soft_br_iterations": soft_iterations,
        "game_value_residual": game_value_residual,
        "game_value_iterations": game_value_iterations,
    }


def paired_summary(rows, environment, method, seeds):
    final = []
    for seed in seeds:
        selected = [r for r in rows if r["environment"] == environment and r["method"] == method and r["seed"] == seed]
        final.append(selected[-1])
    br = np.array([r["hard_br_return"] for r in final], dtype=float)
    exploit = np.array([r["hard_exploitability"] for r in final], dtype=float)
    field_norm = np.array([r["field_norm"] for r in final], dtype=float)
    return {"environment": environment, "method": method, "final_br_mean": float(br.mean()), "final_br_sem": float(br.std(ddof=1) / math.sqrt(len(br))) if len(br) > 1 else 0.0, "final_exploitability": float(exploit.mean()), "final_field_norm": float(field_norm.mean())}


def main() -> None:
    global LR
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tabular", "neural"), required=True)
    parser.add_argument("--environments", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fixed-lr", type=float, default=LR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    LR = args.fixed_lr
    started = time.time()
    catalog = game_catalog()
    default_names = (
        ["RPS", "CyclicControl", "FrequencyHopping"]
        if args.mode == "tabular"
        else [
            "CyclicControl",
            "FrequencyHopping",
            "RoutingInterdiction",
            "SecurityPatrol",
        ]
    )
    names = args.environments or default_names
    games = [catalog[name] for name in names]
    methods = tuple(args.methods) if args.methods else METHODS
    seed_count = args.seeds or (20 if args.mode == "tabular" else 10)
    steps = args.steps or (120 if args.mode == "tabular" else 60)
    if args.smoke:
        seed_count, steps = 1, 2
    default_start = 100 if args.mode == "tabular" else 30
    seed_start = default_start if args.seed_start is None else args.seed_start
    seeds = tuple(range(seed_start, seed_start + seed_count))
    rows = []; diagnostics = []; final_states = {}
    maximum_training_soft_residual = 0.0
    for game in games:
        if float(torch.max(torch.abs(game.transitions.sum(dim=-1) - 1.0))) > 1.0e-12:
            raise RuntimeError("transition sentinel failed for " + game.name)
        print(f"ENVIRONMENT={game.name} MODE={args.mode}", flush=True)
        for seed in seeds:
            z0 = initialize(seed, args.mode, game)
            energy0, gap0, soft0, _ = raw_components(z0, game, args.mode)
            maximum_training_soft_residual = max(maximum_training_soft_residual, soft0)
            normalizers = (max(energy0, 1.0e-10), max(gap0, 1.0e-10))
            for method in methods:
                z = z0.detach().clone(); adam_state = None
                for step in range(steps + 1):
                    if step % CHECKPOINT_EVERY == 0 or step == steps:
                        rows.append({"environment": game.name, "mode": args.mode, "seed": seed, "method": method, "step": step, **metrics(z, game, args.mode)})
                    if step == steps:
                        final_states[(game.name, seed, method)] = z
                        break
                    if method in ("QP+G", "noG"):
                        use_g = method == "QP+G"
                        if use_g:
                            f, g, rotation, cosine = field_and_geometry(z, game, args.mode)
                        else:
                            f = field(z, game, args.mode)
                            g = torch.zeros_like(f)
                            rotation, cosine = float("nan"), float("nan")
                        coefficient = directional_coefficients(z, f, g, game, args.mode, normalizers, use_g)
                        beta, gamma, predicted = solve_box(coefficient, use_g)
                        znew, drift, backtracks, beta_used, gamma_used, safeguard_residual = safeguarded_step(z, f, g, beta, gamma, game, args.mode, normalizers)
                        maximum_training_soft_residual = max(maximum_training_soft_residual, coefficient["max_soft_residual"], safeguard_residual)
                        if use_g:
                            contribution = gamma_used * float(torch.linalg.norm(g)) / max(beta_used * float(torch.linalg.norm(f)) + gamma_used * float(torch.linalg.norm(g)), 1.0e-15)
                            diagnostics.append({"environment": game.name, "mode": args.mode, "seed": seed, "step": step + 1, "rotation": rotation, "cos_fg": cosine, "d": coefficient["d"], "beta": beta_used, "gamma": gamma_used, "gamma_active": float(gamma_used > 1.0e-10), "g_contribution": contribution, "predicted_decrease": -predicted, "realized_decrease": -drift, "inflation": coefficient["inflation"], "backtracks": backtracks, "max_soft_residual": max(coefficient["max_soft_residual"], safeguard_residual)})
                        z = znew
                    else:
                        z, adam_state = classical_step(method, z, game, args.mode, adam_state)
            print(f"  seed {seed} complete", flush=True)

    summaries = [paired_summary(rows, game.name, method, seeds) for game in games for method in methods]
    decisions = []
    if "QP+G" in methods and "noG" in methods:
        for game in games:
            qpg = [r for r in rows if r["environment"] == game.name and r["method"] == "QP+G" and r["step"] == steps]
            nog = [r for r in rows if r["environment"] == game.name and r["method"] == "noG" and r["step"] == steps]
            qpg.sort(key=lambda r: r["seed"]); nog.sort(key=lambda r: r["seed"])
            differences = np.array([a["hard_br_return"] - b["hard_br_return"] for a, b in zip(qpg, nog)])
            if len(differences) > 1:
                critical = float(stats.t.ppf(0.975, len(differences) - 1))
                sem = float(differences.std(ddof=1) / math.sqrt(len(differences)))
            else:
                critical, sem = float("nan"), 0.0
            diag = [r for r in diagnostics if r["environment"] == game.name]
            wins = int(np.sum(differences > 0.0))
            confidence = [float(differences.mean() - critical * sem), float(differences.mean() + critical * sem)] if len(differences) > 1 else [float(differences.mean()), float(differences.mean())]
            sign_p = float(stats.binomtest(wins, len(differences), p=0.5, alternative="greater").pvalue)
            qpg_exploit = float(np.mean([r["hard_exploitability"] for r in qpg])); nog_exploit = float(np.mean([r["hard_exploitability"] for r in nog]))
            decisions.append({"environment": game.name, "all_seed_br_wins": bool(np.all(differences > 0.0)), "br_win_count": wins, "paired_br_gain_mean": float(differences.mean()), "paired_br_gain_95ci": confidence, "one_sided_exact_sign_test_p": sign_p, "qpg_exploitability_mean": qpg_exploit, "nog_exploitability_mean": nog_exploit, "strict_all_seed_positive": bool(np.all(differences > 0.0) and qpg_exploit < nog_exploit), "median_rotation": float(np.median([r["rotation"] for r in diag])) if diag else None, "gamma_activation": float(np.mean([r["gamma_active"] for r in diag])) if diag else None, "mean_g_contribution": float(np.mean([r["g_contribution"] for r in diag])) if diag else None, "inflation_fraction": float(np.mean([r["inflation"] > 1.0e-10 for r in diag])) if diag else None})
        order = sorted(range(len(decisions)), key=lambda index: decisions[index]["one_sided_exact_sign_test_p"])
        running_adjusted = 0.0
        for rank, index in enumerate(order):
            raw_p = decisions[index]["one_sided_exact_sign_test_p"]
            running_adjusted = max(running_adjusted, (len(decisions) - rank) * raw_p)
            decisions[index]["holm_adjusted_sign_p"] = min(1.0, running_adjusted)
        for decision in decisions:
            decision["confirmatory_positive"] = bool(
                decision["paired_br_gain_95ci"][0] > 0.0
                and decision["holm_adjusted_sign_p"] < 0.05
                and decision["qpg_exploitability_mean"] < decision["nog_exploitability_mean"]
            )

    root = Path(__file__).resolve().parent
    output = root / "results" / (f"{args.mode}-exact-gap-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    for filename, data in (("curves.csv", rows), ("diagnostics.csv", diagnostics), ("summary.csv", summaries)):
        if not data:
            continue
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)

    styles = {"QP+G": ("black", "-"), "noG": ("tab:green", "-"), "GDA": ("tab:orange", "--"), "Adam-GDA": ("tab:blue", "--"), "EGM": ("tab:purple", "-."), "PPM-3": ("tab:red", ":")}
    panels = (("hard_br_return", "hard BR return", True), ("hard_exploitability", "hard exploitability", False), ("regularized_gap", "regularized Nash gap", False), ("field_norm", "regularized field norm", False))
    fig, axes = plt.subplots(len(games), 4, figsize=(14.6, 3.2 * len(games)), squeeze=False)
    for row_index, game in enumerate(games):
        for col, (key, title, higher) in enumerate(panels):
            axis = axes[row_index, col]
            method_stats = {}
            for method in methods:
                arrays = []
                for seed in seeds:
                    selected = sorted([r for r in rows if r["environment"] == game.name and r["method"] == method and r["seed"] == seed], key=lambda r: r["step"])
                    arrays.append([r[key] for r in selected])
                data = np.asarray(arrays, dtype=float)
                mean = data.mean(axis=0); sem = data.std(axis=0, ddof=1) / math.sqrt(len(seeds)) if len(seeds) > 1 else np.zeros_like(mean)
                method_stats[method] = (mean, sem)
            stable_pool = [(mean, sem) for method, (mean, sem) in method_stats.items() if method != "Adam-GDA"]
            if not stable_pool:
                stable_pool = list(method_stats.values())
            stable_low = min(float(np.min(mean - sem)) for mean, sem in stable_pool)
            stable_high = max(float(np.max(mean + sem)) for mean, sem in stable_pool)
            margin = max(0.08 * (stable_high - stable_low), 1.0e-6)
            lower, upper = stable_low - margin, stable_high + margin
            x = np.array(sorted({r["step"] for r in rows if r["environment"] == game.name}))
            for method in methods:
                mean, sem = method_stats[method]
                plot_mean = np.clip(mean, lower, upper) if method == "Adam-GDA" else mean
                plot_sem = np.minimum(sem, np.maximum(upper - plot_mean, 0.0)) if method == "Adam-GDA" else sem
                axis.plot(x, plot_mean, label=method, color=styles[method][0], linestyle=styles[method][1], linewidth=1.8)
                axis.fill_between(x, np.maximum(plot_mean - plot_sem, lower), np.minimum(plot_mean + plot_sem, upper), color=styles[method][0], alpha=0.08)
                if method == "Adam-GDA":
                    high = mean > upper; low = mean < lower
                    if np.any(high):
                        axis.scatter(x[high], np.full(np.sum(high), upper), marker="^", s=18, color=styles[method][0], zorder=4)
                    if np.any(low):
                        axis.scatter(x[low], np.full(np.sum(low), lower), marker="v", s=18, color=styles[method][0], zorder=4)
            axis.set_title(f"{game.name}: {title}", fontsize=10)
            axis.set_xlabel("simultaneous joint update")
            axis.set_ylim(lower, upper)
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, 0.002))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.text(0.5, 0.025, "Triangles at an axis boundary denote clipped Adam-GDA means; exact values are retained in summary.csv.", ha="center", fontsize=8)
    fig.savefig(output / f"{args.mode}_markov_games.pdf", bbox_inches="tight")
    fig.savefig(output / f"{args.mode}_markov_games.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    maximum_hard_residual = max(r["hard_br_residual"] for r in rows)
    maximum_game_value_residual = max(r["game_value_residual"] for r in rows)
    minimum_bridge_slack = min(r["bridge_slack"] for r in rows)
    report = {"protocol": {"mode": args.mode, "environments": names, "methods": methods, "seeds": seeds, "steps": steps, "checkpoint_every": CHECKPOINT_EVERY, "simultaneous_updates": True, "warmup": "none", "fixed_lr": LR, "qp_caps": [BETA_MAX, GAMMA_MAX], "ppm_inner": 3, "discount": DISCOUNT, "entropy_tau": ENTROPY_TAU, "performance_merit": "exact entropy-regularized policy-space Nash gap recomputed at every stencil point", "direction": "G=DF F unchanged", "fairness": "same initialization and outer-update count; oracle cost is not matched"}, "maximum_training_soft_br_bellman_residual": maximum_training_soft_residual, "maximum_evaluation_hard_br_bellman_residual": maximum_hard_residual, "maximum_game_value_shapley_residual": maximum_game_value_residual, "minimum_performance_bridge_slack": minimum_bridge_slack, "summaries": summaries, "decisions": decisions, "elapsed_seconds": time.time() - started}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DECISIONS=" + json.dumps(decisions, sort_keys=True), flush=True)
    print("RESULT_DIR=" + str(output), flush=True)


if __name__ == "__main__":
    main()
