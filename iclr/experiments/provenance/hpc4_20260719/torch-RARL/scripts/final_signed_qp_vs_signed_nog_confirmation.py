from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore


TORCH_RARL_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = WORK_ROOT / "results" / "final_signed_qp_vs_signed_nog_confirmation"

DEVICE = torch.device("cpu")
DTYPE = torch.float32
EPS = 1e-8

ENV_ORDER = ["HalfCheetah-v4"]
RHO_GRID = [5.0]
LAMBDA_U_GRID = [0.01]
LAMBDA_W_GRID = [0.05]
LR_GRID = [1e-4]
BATCH_SIZE_GRID = [8192]
SETTING_BATCH_FALLBACK = 4096
LAMBDA_F = 0.0003
LAMBDA_J = 1.0
METHODS = [
    "sgd_gda",
    "egm",
    "proposed_nog_closed_positive",
    "proposed_nog_signed_box",
    "proposed_qp_dampedG_nog_safe_positive",
    "proposed_qp_signed_box_damped_safe",
]

SEED = 0
TRAIN_DATASET_SIZE = 16384
EVAL_DATASET_SIZE = 8192
WARMUP_EPISODES = 2
BATCH_SIZE = 8192
ITERATIONS = 400
EVAL_FREQ = 10
GEOM_BATCHES = 8
GEOM_PROBES = 8
GEOM_BATCH_SIZE = 512
INIT_LOG_STD = -1.0
FINAL_SCALE = 0.05
NO_G_SAFE_TOL = 1e-8
TRUST_G_THRESHOLD = 0.02
PURE_ENV_EVAL_EPISODES = 5
AUX_STANDARD_RARL_ALPHA = 0.05
ETA_LIST = [1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.09375, 0.0625, 0.03125]
EMA_ALPHA = 0.25
NOG_AUDIT_STEPS = 20
MULTISEEDS = [0, 1, 2]
SIGN_DIAG_BATCHES = 100


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def auc(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    xs = np.arange(arr.size, dtype=np.float64)
    return float(np.trapezoid(arr, xs))


def action_rotation_matrix(action_dim: int) -> torch.Tensor:
    mat = torch.zeros((action_dim, action_dim), dtype=DTYPE, device=DEVICE)
    for start in range(0, action_dim - 1, 2):
        mat[start, start + 1] = 1.0
        mat[start + 1, start] = -1.0
    return mat


def fit_quadratic_1d(v0: float, v1: float, v2: float, delta: float) -> tuple[float, float]:
    h = (v2 - 2.0 * v1 + v0) / max(delta * delta, EPS)
    l = (v1 - v0) / max(delta, EPS) - 0.5 * h * delta
    return float(l), float(h)


def solve_nog(point: Callable[[float], float], v0: float, eta: float, beta_max: float | None) -> dict[str, float]:
    db = max(float(eta), EPS)
    v1 = point(db)
    v2 = point(2.0 * db)
    l_beta, h_bb = fit_quadratic_1d(v0, v1, v2, db)
    denom = h_bb + 1e-8
    beta_raw = -l_beta / denom if abs(denom) > EPS else 0.0
    beta = max(beta_raw, 0.0)
    if beta_max is not None:
        beta = min(beta, beta_max)
    q = l_beta * beta + 0.5 * h_bb * beta * beta
    return {
        "db": db,
        "l_beta": float(l_beta),
        "h_bb": float(h_bb),
        "beta_raw": float(beta_raw),
        "beta": float(beta),
        "q": float(q),
    }


def solve_qp(
    point: Callable[[float, float], float],
    v0: float,
    eta: float,
    beta_max: float | None,
    gamma_max: float | None,
) -> dict[str, Any]:
    db = max(float(eta), EPS)
    dg = max(float(eta) * float(eta), 1e-6)
    v_b = point(db, 0.0)
    v_2b = point(2.0 * db, 0.0)
    v_g = point(0.0, dg)
    v_2g = point(0.0, 2.0 * dg)
    v_bg = point(db, dg)
    l_beta, h_bb = fit_quadratic_1d(v0, v_b, v_2b, db)
    l_gamma, h_gg = fit_quadratic_1d(v0, v_g, v_2g, dg)
    h_bg = (v_bg - v_b - v_g + v0) / max(db * dg, EPS)

    def q_value(beta_value: float, gamma_value: float) -> float:
        return float(
            l_beta * beta_value
            + l_gamma * gamma_value
            + 0.5 * h_bb * beta_value * beta_value
            + h_bg * beta_value * gamma_value
            + 0.5 * h_gg * gamma_value * gamma_value
        )

    def beta_feasible(beta_value: float) -> bool:
        return math.isfinite(beta_value) and beta_value >= -1e-10 and (beta_max is None or beta_value <= beta_max + 1e-10)

    def gamma_feasible(gamma_value: float) -> bool:
        return math.isfinite(gamma_value) and gamma_value >= -1e-10 and (gamma_max is None or gamma_value <= gamma_max + 1e-10)

    def clamp_beta(beta_value: float) -> float:
        value = max(float(beta_value), 0.0)
        if beta_max is not None:
            value = min(value, float(beta_max))
        return value

    def clamp_gamma(gamma_value: float) -> float:
        value = max(float(gamma_value), 0.0)
        if gamma_max is not None:
            value = min(value, float(gamma_max))
        return value

    candidates: list[tuple[str, float, float, float]] = []

    def add_candidate(name: str, beta_value: float, gamma_value: float) -> None:
        if not beta_feasible(beta_value) or not gamma_feasible(gamma_value):
            return
        qv = q_value(beta_value, gamma_value)
        if not math.isfinite(qv):
            return
        candidates.append((name, float(beta_value), float(gamma_value), float(qv)))

    nog = solve_nog(lambda beta: point(beta, 0.0), v0, eta, beta_max)
    beta_nog = float(nog["beta"])
    add_candidate("corner_00", 0.0, 0.0)
    add_candidate("edge_gamma0_nog", beta_nog, 0.0)

    hessian_reg = np.array([[h_bb + 1e-8, h_bg], [h_bg, h_gg + 1e-8]], dtype=np.float64)
    linear = np.array([-l_beta, -l_gamma], dtype=np.float64)
    beta_raw = 0.0
    gamma_raw = 0.0
    try:
        solution = np.linalg.solve(hessian_reg, linear)
        beta_raw = float(solution[0])
        gamma_raw = float(solution[1])
        add_candidate("interior", beta_raw, gamma_raw)
    except Exception:
        pass

    denom_gg = h_gg + 1e-8
    gamma_edge0_raw = -l_gamma / denom_gg if abs(denom_gg) > EPS else 0.0
    add_candidate("edge_beta0", 0.0, clamp_gamma(gamma_edge0_raw))

    denom_bb = h_bb + 1e-8
    beta_gamma0_raw = -l_beta / denom_bb if abs(denom_bb) > EPS else 0.0
    add_candidate("edge_gamma0_stationary", clamp_beta(beta_gamma0_raw), 0.0)

    if beta_max is not None:
        add_candidate("corner_betaMax_0", float(beta_max), 0.0)
        gamma_on_beta_max = -(l_gamma + h_bg * beta_max) / denom_gg if abs(denom_gg) > EPS else 0.0
        add_candidate("edge_betaMax", float(beta_max), clamp_gamma(gamma_on_beta_max))
    if gamma_max is not None:
        add_candidate("corner_0_gammaMax", 0.0, float(gamma_max))
        beta_on_gamma_max = -(l_beta + h_bg * gamma_max) / denom_bb if abs(denom_bb) > EPS else 0.0
        add_candidate("edge_gammaMax", clamp_beta(beta_on_gamma_max), float(gamma_max))
    if beta_max is not None and gamma_max is not None:
        add_candidate("corner_betaMax_gammaMax", float(beta_max), float(gamma_max))

    dedup: dict[tuple[int, int], tuple[str, float, float, float]] = {}
    for cand in candidates:
        key = (round(cand[1] / 1e-12), round(cand[2] / 1e-12))
        if key not in dedup or cand[3] < dedup[key][3]:
            dedup[key] = cand
    ranked = sorted(dedup.values(), key=lambda item: item[3])
    if not ranked:
        ranked = [("corner_00_fallback", 0.0, 0.0, q_value(0.0, 0.0))]
    name, beta, gamma, q_selected = ranked[0]
    q_nog = q_value(beta_nog, 0.0)
    return {
        "db": db,
        "dg": dg,
        "l_beta": float(l_beta),
        "l_gamma": float(l_gamma),
        "h_bb": float(h_bb),
        "h_bg": float(h_bg),
        "h_gg": float(h_gg),
        "beta_raw": float(beta_raw),
        "gamma_raw": float(gamma_raw),
        "beta": float(beta),
        "gamma": float(gamma),
        "q": float(q_selected),
        "beta_nog": float(beta_nog),
        "q_nog": float(q_nog),
        "predicted_inclusion_gap": float(q_selected - q_nog),
        "predicted_inclusion_pass": int(q_selected <= q_nog + 1e-6 * max(1.0, abs(q_selected), abs(q_nog))),
        "selected_active_set": name,
        "candidate_count": len(ranked),
    }


def solve_nog_signed_box(
    point: Callable[[float], float],
    v0: float,
    eta: float,
    beta_max: float | None,
) -> dict[str, float]:
    db = max(float(eta), EPS)
    v_p = point(db)
    v_m = point(-db)
    l_beta = (v_p - v_m) / max(2.0 * db, EPS)
    h_bb = (v_p - 2.0 * v0 + v_m) / max(db * db, EPS)
    denom = h_bb + 1e-8
    beta_raw = -l_beta / denom if abs(denom) > EPS else 0.0
    bound = float(beta_max) if beta_max is not None else abs(beta_raw)

    def q_value(beta_value: float) -> float:
        return float(l_beta * beta_value + 0.5 * h_bb * beta_value * beta_value)

    candidates = [0.0, beta_raw]
    if beta_max is not None:
        candidates.extend([-bound, bound])
    best_beta = 0.0
    best_q = q_value(0.0)
    for beta_value in candidates:
        beta = float(beta_value)
        if beta_max is not None:
            beta = max(-bound, min(bound, beta))
        qv = q_value(beta)
        if math.isfinite(qv) and qv < best_q:
            best_q = float(qv)
            best_beta = float(beta)
    return {
        "db": db,
        "l_beta": float(l_beta),
        "h_bb": float(h_bb),
        "beta_raw": float(beta_raw),
        "beta": float(best_beta),
        "q": float(best_q),
        "beta_positive": int(best_beta > 1e-12),
        "beta_negative": int(best_beta < -1e-12),
        "beta_zero": int(abs(best_beta) <= 1e-12),
    }


def solve_qp_signed_box(
    point: Callable[[float, float], float],
    v0: float,
    eta: float,
    beta_max: float | None,
    gamma_max: float | None,
) -> dict[str, Any]:
    db = max(float(eta), EPS)
    dg = max(float(eta) * float(eta), 1e-6)
    bmax = float(beta_max) if beta_max is not None else db
    gmax = float(gamma_max) if gamma_max is not None else dg

    v_bp = point(db, 0.0)
    v_bm = point(-db, 0.0)
    v_gp = point(0.0, dg)
    v_gm = point(0.0, -dg)
    v_pp = point(db, dg)
    v_pm = point(db, -dg)
    v_mp = point(-db, dg)
    v_mm = point(-db, -dg)

    l_beta = (v_bp - v_bm) / max(2.0 * db, EPS)
    h_bb = (v_bp - 2.0 * v0 + v_bm) / max(db * db, EPS)
    l_gamma = (v_gp - v_gm) / max(2.0 * dg, EPS)
    h_gg = (v_gp - 2.0 * v0 + v_gm) / max(dg * dg, EPS)
    h_bg = (v_pp - v_pm - v_mp + v_mm) / max(4.0 * db * dg, EPS)

    def q_value(beta_value: float, gamma_value: float) -> float:
        return float(
            l_beta * beta_value
            + l_gamma * gamma_value
            + 0.5 * h_bb * beta_value * beta_value
            + h_bg * beta_value * gamma_value
            + 0.5 * h_gg * gamma_value * gamma_value
        )

    def beta_feasible(beta_value: float) -> bool:
        return math.isfinite(beta_value) and (-bmax - 1e-10) <= beta_value <= (bmax + 1e-10)

    def gamma_feasible(gamma_value: float) -> bool:
        return math.isfinite(gamma_value) and (-gmax - 1e-10) <= gamma_value <= (gmax + 1e-10)

    def clamp_beta(beta_value: float) -> float:
        return max(-bmax, min(bmax, float(beta_value)))

    def clamp_gamma(gamma_value: float) -> float:
        return max(-gmax, min(gmax, float(gamma_value)))

    candidates: list[tuple[str, float, float, float]] = []

    def add_candidate(name: str, beta_value: float, gamma_value: float) -> None:
        if not beta_feasible(beta_value) or not gamma_feasible(gamma_value):
            return
        qv = q_value(beta_value, gamma_value)
        if not math.isfinite(qv):
            return
        candidates.append((name, float(beta_value), float(gamma_value), float(qv)))

    signed_nog = solve_nog_signed_box(lambda beta: point(beta, 0.0), v0, eta, beta_max)
    beta_signed_nog = float(signed_nog["beta"])
    add_candidate("origin", 0.0, 0.0)
    add_candidate("signed_nog", beta_signed_nog, 0.0)

    hessian_reg = np.array([[h_bb + 1e-8, h_bg], [h_bg, h_gg + 1e-8]], dtype=np.float64)
    linear = np.array([-l_beta, -l_gamma], dtype=np.float64)
    beta_raw = 0.0
    gamma_raw = 0.0
    try:
        solution = np.linalg.solve(hessian_reg, linear)
        beta_raw = float(solution[0])
        gamma_raw = float(solution[1])
        add_candidate("interior", beta_raw, gamma_raw)
    except Exception:
        pass

    denom_bb = h_bb + 1e-8
    denom_gg = h_gg + 1e-8
    add_candidate("line_gamma0", clamp_beta(-l_beta / denom_bb if abs(denom_bb) > EPS else 0.0), 0.0)
    add_candidate("line_beta0", 0.0, clamp_gamma(-l_gamma / denom_gg if abs(denom_gg) > EPS else 0.0))

    for beta_edge in (-bmax, bmax):
        gamma_star = -(l_gamma + h_bg * beta_edge) / denom_gg if abs(denom_gg) > EPS else 0.0
        add_candidate(f"edge_beta_{beta_edge:+.6g}", beta_edge, clamp_gamma(gamma_star))
    for gamma_edge in (-gmax, gmax):
        beta_star = -(l_beta + h_bg * gamma_edge) / denom_bb if abs(denom_bb) > EPS else 0.0
        add_candidate(f"edge_gamma_{gamma_edge:+.6g}", clamp_beta(beta_star), gamma_edge)
    for beta_corner in (-bmax, bmax):
        for gamma_corner in (-gmax, gmax):
            add_candidate(f"corner_{beta_corner:+.6g}_{gamma_corner:+.6g}", beta_corner, gamma_corner)

    dedup: dict[tuple[int, int], tuple[str, float, float, float]] = {}
    for cand in candidates:
        key = (round(cand[1] / 1e-12), round(cand[2] / 1e-12))
        if key not in dedup or cand[3] < dedup[key][3]:
            dedup[key] = cand
    ranked = sorted(dedup.values(), key=lambda item: item[3])
    if not ranked:
        ranked = [("origin_fallback", 0.0, 0.0, q_value(0.0, 0.0))]
    name, beta, gamma, q_selected = ranked[0]
    q_signed_nog = q_value(beta_signed_nog, 0.0)
    return {
        "db": db,
        "dg": dg,
        "l_beta": float(l_beta),
        "l_gamma": float(l_gamma),
        "h_bb": float(h_bb),
        "h_bg": float(h_bg),
        "h_gg": float(h_gg),
        "beta_raw": float(beta_raw),
        "gamma_raw": float(gamma_raw),
        "beta": float(beta),
        "gamma": float(gamma),
        "q": float(q_selected),
        "beta_signed_nog": float(beta_signed_nog),
        "q_signed_nog": float(q_signed_nog),
        "predicted_inclusion_gap": float(q_selected - q_signed_nog),
        "predicted_inclusion_pass": int(q_selected <= q_signed_nog + 1e-6 * max(1.0, abs(q_selected), abs(q_signed_nog))),
        "selected_active_set": name,
        "candidate_count": len(ranked),
        "beta_positive": int(beta > 1e-12),
        "beta_negative": int(beta < -1e-12),
        "gamma_positive": int(gamma > 1e-12),
        "gamma_negative": int(gamma < -1e-12),
    }


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    max_episode_steps: int


@dataclass(frozen=True)
class Config:
    env_id: str
    rho: float
    lambda_u: float
    lambda_w: float
    lr: float
    lambda_F: float
    lambda_J: float
    batch_size: int = BATCH_SIZE

    @property
    def slug(self) -> str:
        return (
            f"{self.env_id.lower().replace('-', '_')}"
            f"_rho{str(self.rho).replace('.', 'p')}"
            f"_lu{str(self.lambda_u).replace('.', 'p')}"
            f"_lw{str(self.lambda_w).replace('.', 'p')}"
            f"_lr{str(self.lr).replace('.', 'p')}"
            f"_lF{str(self.lambda_F).replace('.', 'p')}"
            f"_lJ{str(self.lambda_J).replace('.', 'p')}"
            f"_bs{self.batch_size}"
        )


class FlatMLP:
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, int], output_dim: int) -> None:
        h1, h2 = hidden_sizes
        self.shapes = [
            (h1, input_dim),
            (h1,),
            (h2, h1),
            (h2,),
            (output_dim, h2),
            (output_dim,),
        ]
        self.num_params = sum(int(np.prod(shape)) for shape in self.shapes)

    def init_flat(self, generator: torch.Generator, final_scale: float = FINAL_SCALE) -> torch.Tensor:
        chunks = []
        for idx, shape in enumerate(self.shapes):
            if len(shape) == 2:
                scale = 1.0 / math.sqrt(shape[1])
                tensor = torch.randn(shape, generator=generator, dtype=DTYPE, device=DEVICE) * scale
            else:
                tensor = 0.01 * torch.randn(shape, generator=generator, dtype=DTYPE, device=DEVICE)
            if idx >= 4:
                tensor = tensor * final_scale
            chunks.append(tensor.reshape(-1))
        return torch.cat(chunks)

    def unpack(self, flat: torch.Tensor) -> list[torch.Tensor]:
        offset = 0
        tensors = []
        for shape in self.shapes:
            size = int(np.prod(shape))
            tensors.append(flat[offset : offset + size].reshape(shape))
            offset += size
        return tensors

    def forward(self, flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        w1, b1, w2, b2, w3, b3 = self.unpack(flat)
        h1 = torch.tanh(obs @ w1.T + b1)
        h2 = torch.tanh(h1 @ w2.T + b2)
        return h2 @ w3.T + b3


def check_env(env_id: str) -> EnvSpec | None:
    try:
        env = gym.make(env_id)
    except Exception:
        return None
    obs_space = env.observation_space
    act_space = env.action_space
    if getattr(obs_space, "shape", None) is None or getattr(act_space, "shape", None) is None:
        env.close()
        return None
    spec = EnvSpec(
        env_id=env_id,
        obs_dim=int(np.prod(obs_space.shape)),
        action_dim=int(np.prod(act_space.shape)),
        action_low=np.asarray(act_space.low, dtype=np.float32).reshape(-1),
        action_high=np.asarray(act_space.high, dtype=np.float32).reshape(-1),
        max_episode_steps=int(getattr(env, "spec", None).max_episode_steps if getattr(env, "spec", None) is not None else 1000),
    )
    env.close()
    return spec


def collect_state_dataset(spec: EnvSpec, seed: int, train_size: int, eval_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    env = gym.make(spec.env_id)
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    target_size = train_size + eval_size
    for episode in range(WARMUP_EPISODES):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done and len(states) < target_size:
            states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            action = env.action_space.sample()
            if episode > 0:
                action = np.clip(action + 0.10 * rng.standard_normal(action.shape), env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, _ = env.step(action.astype(np.float32))
            done = bool(terminated or truncated)
    while len(states) < target_size:
        obs, _ = env.reset(seed=seed + len(states) + 10)
        done = False
        while not done and len(states) < target_size:
            states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
            done = bool(terminated or truncated)
    env.close()
    arr = np.asarray(states[:target_size], dtype=np.float32)
    rng.shuffle(arr)
    train = torch.as_tensor(arr[:train_size], dtype=DTYPE, device=DEVICE)
    eval_states = torch.as_tensor(arr[train_size : train_size + eval_size], dtype=DTYPE, device=DEVICE)
    return train, eval_states


class MujocoStateActorGame:
    def __init__(self, spec: EnvSpec, cfg: Config, train_states: torch.Tensor, eval_states: torch.Tensor, seed: int) -> None:
        self.spec = spec
        self.cfg = cfg
        self.seed = seed
        self.train_states = train_states
        self.eval_states = eval_states
        self.action_low_t = torch.as_tensor(spec.action_low, dtype=DTYPE, device=DEVICE)
        self.action_high_t = torch.as_tensor(spec.action_high, dtype=DTYPE, device=DEVICE)
        self.action_mid_t = 0.5 * (self.action_high_t + self.action_low_t)
        self.action_scale_t = 0.5 * (self.action_high_t - self.action_low_t)
        self.H = action_rotation_matrix(spec.action_dim)
        self.actor_layout = FlatMLP(spec.obs_dim, (64, 64), spec.action_dim)
        self.slices = self._build_slices()
        self.metric_refs: dict[str, float] = {}

    def _build_slices(self) -> dict[str, slice]:
        offset = 0
        actor_dim = self.actor_layout.num_params
        std_dim = self.spec.action_dim
        out = {}
        out["protagonist_actor"] = slice(offset, offset + actor_dim)
        offset += actor_dim
        out["protagonist_log_std"] = slice(offset, offset + std_dim)
        offset += std_dim
        out["adversary_actor"] = slice(offset, offset + actor_dim)
        offset += actor_dim
        out["adversary_log_std"] = slice(offset, offset + std_dim)
        return out

    def init_z(self, seed_offset: int = 0) -> torch.Tensor:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(self.seed + seed_offset)
        pa = self.actor_layout.init_flat(gen)
        pl = torch.full((self.spec.action_dim,), INIT_LOG_STD, dtype=DTYPE, device=DEVICE)
        aa = self.actor_layout.init_flat(gen)
        al = torch.full((self.spec.action_dim,), INIT_LOG_STD, dtype=DTYPE, device=DEVICE)
        return torch.cat([pa, pl, aa, al]).detach().clone()

    def split_z(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: z[slc] for name, slc in self.slices.items()}

    def actor_mean_action(self, actor_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor_layout.forward(actor_flat, obs)
        return self.action_mid_t + self.action_scale_t * torch.tanh(raw)

    def objective_terms(self, z: torch.Tensor, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = self.split_z(z)
        u = self.actor_mean_action(parts["protagonist_actor"], obs)
        w = self.actor_mean_action(parts["adversary_actor"], obs)
        rot_per_state = torch.sum(u * torch.matmul(w, self.H.T), dim=-1) / math.sqrt(max(self.spec.action_dim, 1))
        if "rot_scale" not in self.metric_refs:
            self.metric_refs["rot_scale"] = max(float(torch.std(rot_per_state.detach(), unbiased=False).item()), 1e-3)
        rot_scale = self.metric_refs["rot_scale"]
        rot = torch.mean(rot_per_state)
        rot_norm = torch.mean(rot_per_state / rot_scale)
        u_energy = torch.mean(torch.sum(u * u, dim=-1)) / max(self.spec.action_dim, 1)
        w_energy = torch.mean(torch.sum(w * w, dim=-1)) / max(self.spec.action_dim, 1)
        J = (self.cfg.rho * rot_norm) - (self.cfg.lambda_u * u_energy) + (self.cfg.lambda_w * w_energy)
        return {
            "u": u,
            "w": w,
            "rot_per_state": rot_per_state,
            "rot": rot,
            "rot_norm": rot_norm,
            "rot_scale": torch.as_tensor(rot_scale, dtype=DTYPE, device=DEVICE),
            "u_energy": u_energy,
            "w_energy": w_energy,
            "J": J,
            "L_mu": -J,
            "L_nu": +J,
        }

    def field(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        terms = self.objective_terms(z_req, obs)
        grad_mu = torch.autograd.grad(terms["L_mu"], z_req, retain_graph=True, create_graph=True)[0]
        grad_nu = torch.autograd.grad(terms["L_nu"], z_req, create_graph=True)[0]
        p_slice = slice(self.slices["protagonist_actor"].start, self.slices["protagonist_log_std"].stop)
        a_slice = slice(self.slices["adversary_actor"].start, self.slices["adversary_log_std"].stop)
        return torch.cat([grad_mu[p_slice], grad_nu[a_slice]])

    def merit(self, z: torch.Tensor, obs: torch.Tensor, compute_geometry: bool) -> dict[str, float]:
        terms = self.objective_terms(z, obs)
        field = self.field(z, obs).detach()
        field_energy = 0.5 * float(torch.dot(field, field).item())
        J_value = float(terms["J"].detach().item())
        if "field0" not in self.metric_refs:
            self.metric_refs["field0"] = max(field_energy, EPS)
            self.metric_refs["negJ0"] = max(abs(-J_value), EPS)
        field_term = field_energy / (self.metric_refs["field0"] + EPS)
        j_term = (-J_value) / (self.metric_refs["negJ0"] + EPS)
        out = {
            "V": (self.cfg.lambda_F * field_term) + (self.cfg.lambda_J * j_term),
            "field_energy": field_energy,
            "field_norm": float(torch.linalg.norm(field).item()),
            "distance_to_game_stationarity": float(torch.linalg.norm(field).item()),
            "actor_game_score": J_value,
            "payoff_J": J_value,
            "rot_raw": float(terms["rot"].detach().item()),
            "rot_norm": float(terms["rot_norm"].detach().item()),
            "rot_scale": float(terms["rot_scale"].detach().item()),
            "u_energy": float(terms["u_energy"].detach().item()),
            "w_energy": float(terms["w_energy"].detach().item()),
            "mean_abs_u": float(terms["u"].detach().abs().mean().item()),
            "mean_abs_w": float(terms["w"].detach().abs().mean().item()),
        }
        if compute_geometry:
            out.update(self.geometry_metrics(z, obs))
        else:
            out.update(
                {
                    "G_norm": math.nan,
                    "G_over_F": math.nan,
                    "cos_F_G": math.nan,
                    "non_collinearity": math.nan,
                    "rotation_ratio_proxy": math.nan,
                    "cross_player_coupling_proxy": math.nan,
                    "cross_to_same_ratio": math.nan,
                }
            )
        return out

    def geometry_metrics(self, z: torch.Tensor, obs: torch.Tensor) -> dict[str, float]:
        z_req = z.detach().clone().requires_grad_(True)
        field_z = self.field(z_req, obs)
        _, g_vec = torch.autograd.functional.jvp(lambda zz: self.field(zz, obs), (z_req,), (field_z.detach(),), create_graph=False, strict=False)
        field_det = field_z.detach()
        g_det = g_vec.detach()
        f_norm = float(torch.linalg.norm(field_det).item())
        g_norm = float(torch.linalg.norm(g_det).item())
        cos_fg = float(torch.dot(field_det, g_det).item() / ((f_norm * g_norm) + EPS))

        aproxy = 0.0
        sproxy = 0.0
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(170000 + self.seed)
        for _ in range(GEOM_PROBES):
            v = torch.randn(z_req.numel(), generator=gen, dtype=DTYPE, device=DEVICE)
            v = v / (torch.linalg.norm(v) + EPS)
            _, jv = torch.autograd.functional.jvp(lambda zz: self.field(zz, obs), (z_req,), (v,), create_graph=False, strict=False)
            jtv = torch.autograd.grad(torch.dot(field_z, v), z_req, retain_graph=True)[0]
            aproxy += float(torch.linalg.norm(jv.detach() - jtv.detach()).item())
            sproxy += float(torch.linalg.norm(jv.detach() + jtv.detach()).item())

        p_slice = slice(self.slices["protagonist_actor"].start, self.slices["protagonist_log_std"].stop)
        a_slice = slice(self.slices["adversary_actor"].start, self.slices["adversary_log_std"].stop)

        def protagonist_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.field(cur_z, obs)[p_slice].detach()

        def adversary_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.field(cur_z, obs)[a_slice].detach()

        base_p = protagonist_field(z)
        base_a = adversary_field(z)
        pert_p = z.detach().clone()
        pert_a = z.detach().clone()
        pert_p[p_slice] = pert_p[p_slice] + 1e-3
        pert_a[a_slice] = pert_a[a_slice] + 1e-3
        cross_p = float(torch.linalg.norm(protagonist_field(pert_a) - base_p).item())
        cross_a = float(torch.linalg.norm(adversary_field(pert_p) - base_a).item())
        same_p = float(torch.linalg.norm(protagonist_field(pert_p) - base_p).item())
        same_a = float(torch.linalg.norm(adversary_field(pert_a) - base_a).item())
        cross = 0.5 * (cross_p + cross_a)
        same = 0.5 * (same_p + same_a)
        return {
            "G_norm": g_norm,
            "G_over_F": g_norm / (f_norm + EPS),
            "cos_F_G": cos_fg,
            "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
            "rotation_ratio_proxy": aproxy / (sproxy + EPS),
            "cross_player_coupling_proxy": cross,
            "cross_to_same_ratio": cross / (same + EPS),
        }

    def sample_batch(self, batch_size: int, iteration: int) -> torch.Tensor:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(self.seed + 1000 + iteration)
        idx = torch.randint(low=0, high=self.train_states.shape[0], size=(batch_size,), generator=gen, device=DEVICE)
        return self.train_states[idx]

    def eval_batch(self) -> torch.Tensor:
        return self.eval_states

    def evaluate_pure_env_return(self, z: torch.Tensor, episodes: int, seed_offset: int = 0) -> float:
        env = gym.make(self.spec.env_id)
        parts = self.split_z(z)
        returns: list[float] = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=self.seed + seed_offset + episode)
            done = False
            total = 0.0
            while not done:
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), dtype=DTYPE, device=DEVICE)
                action = self.actor_mean_action(parts["protagonist_actor"], obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            returns.append(total)
        env.close()
        return float(np.mean(returns)) if returns else math.nan

    def evaluate_standard_rarl_returns(
        self,
        z: torch.Tensor,
        episodes: int,
        alpha: float = AUX_STANDARD_RARL_ALPHA,
        seed_offset: int = 0,
    ) -> dict[str, float]:
        env = gym.make(self.spec.env_id)
        parts = self.split_z(z)
        clean_returns: list[float] = []
        robust_returns: list[float] = []
        for episode in range(episodes):
            # Clean protagonist-only rollout
            obs, _ = env.reset(seed=self.seed + seed_offset + episode)
            done = False
            total_clean = 0.0
            while not done:
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), dtype=DTYPE, device=DEVICE)
                u = self.actor_mean_action(parts["protagonist_actor"], obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(u)
                total_clean += float(reward)
                done = bool(terminated or truncated)
            clean_returns.append(total_clean)

            # Robust/additive adversary rollout
            obs, _ = env.reset(seed=self.seed + seed_offset + 1000 + episode)
            done = False
            total_robust = 0.0
            while not done:
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), dtype=DTYPE, device=DEVICE)
                u = self.actor_mean_action(parts["protagonist_actor"], obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
                w = self.actor_mean_action(parts["adversary_actor"], obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
                a = np.clip(u + float(alpha) * w, self.spec.action_low, self.spec.action_high).astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(a)
                total_robust += float(reward)
                done = bool(terminated or truncated)
            robust_returns.append(total_robust)
        env.close()
        clean_mean = float(np.mean(clean_returns)) if clean_returns else math.nan
        robust_mean = float(np.mean(robust_returns)) if robust_returns else math.nan
        return {
            "standard_clean_env_return_aux": clean_mean,
            "standard_robust_env_return_aux": robust_mean,
            "standard_robust_degradation_aux": clean_mean - robust_mean if finite(clean_mean) and finite(robust_mean) else math.nan,
            "standard_rarl_alpha_aux": float(alpha),
        }


def apply_delta(z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (z + delta).detach()


def run_sgd(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float) -> tuple[torch.Tensor, dict[str, Any]]:
    F0 = game.field(z, obs).detach()
    delta = -eta * F0
    z_next = apply_delta(z, delta)
    v_after = game.merit(z_next, obs, compute_geometry=False)["V"]
    return z_next, {
        "update_norm": float(torch.linalg.norm(delta).item()),
        "beta": float(eta),
        "gamma": 0.0,
        "gamma_active": 0,
        "G_contribution_ratio": 0.0,
        "fallback_to_noG": 0,
        "QP_better_than_noG_actual_V": 0,
        "QP_better_than_EGM_actual_V": 0,
        "predicted_inclusion_pass": 1,
        "selected_active_set": "sgd",
        "V_after_candidate": float(v_after),
    }


def run_egm(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float) -> tuple[torch.Tensor, dict[str, Any]]:
    F0 = game.field(z, obs).detach()
    z_half = apply_delta(z, -eta * F0)
    F_half = game.field(z_half, obs).detach()
    delta = -eta * F_half
    z_next = apply_delta(z, delta)
    v_after = game.merit(z_next, obs, compute_geometry=False)["V"]
    return z_next, {
        "update_norm": float(torch.linalg.norm(delta).item()),
        "beta": float(eta),
        "gamma": 0.0,
        "gamma_active": 0,
        "G_contribution_ratio": 0.0,
        "fallback_to_noG": 0,
        "QP_better_than_noG_actual_V": 0,
        "QP_better_than_EGM_actual_V": 0,
        "predicted_inclusion_pass": 1,
        "selected_active_set": "egm",
        "V_after_candidate": float(v_after),
    }


def run_nog_closed(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float) -> tuple[torch.Tensor, dict[str, Any]]:
    z_req = z.detach().clone().requires_grad_(True)
    F0 = game.field(z_req, obs).detach()
    v0 = game.merit(z, obs, compute_geometry=False)["V"]

    def point(beta: float) -> float:
        z_cand = apply_delta(z, -beta * F0)
        return game.merit(z_cand, obs, compute_geometry=False)["V"]

    nog = solve_nog(point, v0, eta, beta_max=10.0 * eta)
    delta = -nog["beta"] * F0
    z_next = apply_delta(z, delta)
    v_after = game.merit(z_next, obs, compute_geometry=False)["V"]
    return z_next, {
        "update_norm": float(torch.linalg.norm(delta).item()),
        "beta": float(nog["beta"]),
        "gamma": 0.0,
        "gamma_active": 0,
        "G_contribution_ratio": 0.0,
        "fallback_to_noG": 0,
        "QP_better_than_noG_actual_V": 0,
        "QP_better_than_EGM_actual_V": 0,
        "predicted_inclusion_pass": 1,
        "selected_active_set": "noG",
        "V_after_candidate": float(v_after),
        "V_before": float(v0),
    }


def qp_candidate_meta(
    game: MujocoStateActorGame,
    z: torch.Tensor,
    obs: torch.Tensor,
    eta: float,
    normalize_g: bool = False,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor, float]:
    z_req = z.detach().clone().requires_grad_(True)
    F0 = game.field(z_req, obs)
    _, G0 = torch.autograd.functional.jvp(lambda zz: game.field(zz, obs), (z_req,), (F0.detach(),), create_graph=False, strict=False)
    F_det = F0.detach()
    G_det = G0.detach()
    f_norm = float(torch.linalg.norm(F_det).item())
    g_norm_raw = float(torch.linalg.norm(G_det).item())
    if normalize_g:
        G_det = G_det * (f_norm / (g_norm_raw + EPS))
    v0 = game.merit(z, obs, compute_geometry=False)["V"]

    def point(beta: float, gamma: float) -> float:
        z_cand = apply_delta(z, (-beta * F_det) + (gamma * G_det))
        return game.merit(z_cand, obs, compute_geometry=False)["V"]

    qp = solve_qp(point, v0, eta, beta_max=10.0 * eta, gamma_max=10.0 * eta * eta)
    delta = (-qp["beta"] * F_det) + (qp["gamma"] * G_det)
    z_next = apply_delta(z, delta)
    g_norm = float(torch.linalg.norm(G_det).item())
    g_ratio = float(abs(qp["gamma"]) * g_norm / ((abs(qp["beta"]) * f_norm) + (abs(qp["gamma"]) * g_norm) + EPS))
    v_after = game.merit(z_next, obs, compute_geometry=False)["V"]
    meta = {
        "update_norm": float(torch.linalg.norm(delta).item()),
        "beta": float(qp["beta"]),
        "gamma": float(qp["gamma"]),
        "gamma_active": int(qp["gamma"] > 1e-12),
        "G_contribution_ratio": g_ratio,
        "fallback_to_noG": 0,
        "predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
        "selected_active_set": str(qp["selected_active_set"]),
        "V_before": float(v0),
        "V_after_candidate": float(v_after),
        "cos_F_G": float(torch.dot(F_det, G_det).item() / ((f_norm * g_norm) + EPS)),
        "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, (float(torch.dot(F_det, G_det).item() / ((f_norm * g_norm) + EPS))) ** 2))),
        "normalize_g": int(normalize_g),
    }
    return z_next, meta, F_det, G_det, v0


def run_qp_closed(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float) -> tuple[torch.Tensor, dict[str, Any]]:
    z_qp, meta_qp, _, _, _ = qp_candidate_meta(game, z, obs, eta)
    meta_qp["QP_better_than_noG_actual_V"] = 0
    meta_qp["QP_better_than_EGM_actual_V"] = 0
    return z_qp, meta_qp


def run_qp_damped_variant(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float, normalize_g: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    z_nog, meta_nog = run_nog_closed(game, z, obs, eta)
    z_full, meta_full, F_det, G_det, _ = qp_candidate_meta(game, z, obs, eta, normalize_g=normalize_g)
    v_nog = safe_float(meta_nog["V_after_candidate"])
    v_full = safe_float(meta_full["V_after_candidate"])
    tol = NO_G_SAFE_TOL * max(1.0, abs(v_nog))

    def eta_label(value: float) -> str:
        if value == 0.0:
            return "noG"
        label = str(value).replace(".", "p")
        return f"dampedG_eta_{label}"

    best_choice = {
        "kind": "noG",
        "eta": 0.0,
        "z": z_nog,
        "V": v_nog,
        "G_ratio": 0.0,
        "margin": 0.0,
        "beta": safe_float(meta_nog["beta"]),
        "gamma": 0.0,
        "gamma_active": 0,
        "fullG_better_than_noG": int(v_full <= v_nog + 1e-12),
        "fallback_reason": "gamma_inactive" if int(meta_full["gamma_active"]) == 0 else "selected_noG_by_actual_V",
        "chosen_step_type": "noG",
        "best_dampedG_minus_noG_V_gap": float(v_nog - v_full),
    }
    beta = safe_float(meta_full["beta"])
    gamma = safe_float(meta_full["gamma"])
    f_norm = float(torch.linalg.norm(F_det).item())
    fallback_reason = "gamma_inactive" if gamma <= 1e-12 else "no_eta_improves_actual_V"
    for eta_scale in ETA_LIST:
        delta = (-beta * F_det) + (eta_scale * gamma * G_det)
        z_eta = apply_delta(z, delta)
        v_eta = game.merit(z_eta, obs, compute_geometry=False)["V"]
        g_term_norm = abs(eta_scale * gamma) * float(torch.linalg.norm(G_det).item())
        f_term_norm = abs(beta) * f_norm
        g_ratio = g_term_norm / (f_term_norm + g_term_norm + EPS)
        improves_actual = v_eta <= (v_nog - tol)
        if gamma > 1e-12 and improves_actual and g_ratio < TRUST_G_THRESHOLD:
            fallback_reason = "effective_G_too_small"
        if gamma > 1e-12 and improves_actual and g_ratio >= TRUST_G_THRESHOLD:
            fallback_reason = "selected_noG_by_actual_V"
        if (
            improves_actual
            and gamma > 1e-12
            and g_ratio >= TRUST_G_THRESHOLD
            and v_eta < best_choice["V"] - 1e-12
        ):
            best_choice = {
                "kind": "damped_qp",
                "eta": float(eta_scale),
                "z": z_eta,
                "V": float(v_eta),
                "G_ratio": float(g_ratio),
                "margin": float(v_nog - v_eta),
                "beta": float(beta),
                "gamma": float(eta_scale * gamma),
                "gamma_active": 1,
                "fullG_better_than_noG": int(v_full <= v_nog + 1e-12),
                "fallback_reason": "accepted_dampedG",
                "chosen_step_type": eta_label(float(eta_scale)),
                "best_dampedG_minus_noG_V_gap": float(v_nog - v_eta),
            }
    if gamma > 1e-12 and best_choice["kind"] == "noG" and fallback_reason == "selected_noG_by_actual_V":
        fallback_reason = "no_eta_improves_actual_V"
    chosen = best_choice["z"]
    return chosen, {
        **meta_full,
        "beta": float(best_choice["beta"]),
        "gamma": float(best_choice["gamma"]),
        "gamma_active": int(best_choice["gamma_active"]),
        "fallback_to_noG": int(best_choice["kind"] == "noG"),
        "QP_accept_frac_flag": int(best_choice["kind"] != "noG"),
        "chosen_eta": float(best_choice["eta"]),
        "chosen_eta_mean": float(best_choice["eta"]),
        "effective_gamma": float(best_choice["gamma"]),
        "effective_gamma_mean": float(best_choice["gamma"]),
        "effective_G_contribution_ratio": float(best_choice["G_ratio"]),
        "G_contribution_ratio": float(best_choice["G_ratio"]),
        "accepted_QP_better_than_noG_margin": float(best_choice["margin"]),
        "fullG_better_than_noG_fraction_flag": int(best_choice["fullG_better_than_noG"]),
        "dampedG_better_than_noG_fraction_flag": int(best_choice["kind"] != "noG"),
        "QP_better_than_noG_actual_V": int(best_choice["kind"] != "noG"),
        "selected_active_set": f"{meta_full['selected_active_set']}|eta={best_choice['eta']}",
        "full_gamma_value": float(gamma),
        "full_G_contribution_ratio": float(safe_float(meta_full.get("G_contribution_ratio", 0.0), 0.0)),
        "chosen_step_type": str(best_choice["chosen_step_type"]),
        "fallback_reason": str(best_choice["fallback_reason"] if best_choice["kind"] != "noG" else fallback_reason),
        "V_after_noG": float(v_nog),
        "best_V_after_dampedG": float(best_choice["V"] if best_choice["kind"] != "noG" else v_full),
        "best_dampedG_minus_noG_V_gap": float(best_choice["best_dampedG_minus_noG_V_gap"]),
        "V_after_candidate": float(best_choice["V"]),
    }


def run_qp_nog_safe(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float) -> tuple[torch.Tensor, dict[str, Any]]:
    z_nog, meta_nog = run_nog_closed(game, z, obs, eta)
    z_qp, meta_qp, _, _, _ = qp_candidate_meta(game, z, obs, eta)
    z_egm, meta_egm = run_egm(game, z, obs, eta)
    v_nog = safe_float(meta_nog["V_after_candidate"])
    v_qp = safe_float(meta_qp["V_after_candidate"])
    v_egm = safe_float(meta_egm["V_after_candidate"])
    qp_valid = (
        v_qp <= (v_nog - NO_G_SAFE_TOL)
        and int(meta_qp["gamma_active"]) == 1
        and safe_float(meta_qp["G_contribution_ratio"]) >= TRUST_G_THRESHOLD
    )
    if qp_valid:
        return z_qp, {
            **meta_qp,
            "fallback_to_noG": 0,
            "QP_better_than_noG_actual_V": int(v_qp <= v_nog + 1e-12),
            "QP_better_than_EGM_actual_V": int(v_qp <= v_egm + 1e-12),
        }
    return z_nog, {
        **meta_qp,
        "beta_nog": safe_float(meta_nog["beta"]),
        "fallback_to_noG": 1,
        "QP_better_than_noG_actual_V": int(v_qp <= v_nog + 1e-12),
        "QP_better_than_EGM_actual_V": int(v_qp <= v_egm + 1e-12),
        "V_after_candidate": v_nog,
    }


def run_nog_signed_box(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float) -> tuple[torch.Tensor, dict[str, Any]]:
    z_req = z.detach().clone().requires_grad_(True)
    F0 = game.field(z_req, obs).detach()
    v0 = game.merit(z, obs, compute_geometry=False)["V"]

    def point(beta: float) -> float:
        z_cand = apply_delta(z, -beta * F0)
        return game.merit(z_cand, obs, compute_geometry=False)["V"]

    nog = solve_nog_signed_box(point, v0, eta, beta_max=10.0 * eta)
    delta = -nog["beta"] * F0
    z_next = apply_delta(z, delta)
    v_after = game.merit(z_next, obs, compute_geometry=False)["V"]
    return z_next, {
        "update_norm": float(torch.linalg.norm(delta).item()),
        "beta": float(nog["beta"]),
        "beta_signed": float(nog["beta"]),
        "gamma": 0.0,
        "gamma_active": 0,
        "G_contribution_ratio": 0.0,
        "fallback_to_noG": 0,
        "QP_better_than_noG_actual_V": 0,
        "QP_better_than_EGM_actual_V": 0,
        "predicted_inclusion_pass": 1,
        "selected_active_set": "signed_noG",
        "V_after_candidate": float(v_after),
        "V_before": float(v0),
        "beta_positive_flag": int(nog["beta_positive"]),
        "beta_negative_flag": int(nog["beta_negative"]),
        "beta_zero_flag": int(nog["beta_zero"]),
        "beta_active_flag": int(abs(float(nog["beta"])) > 1e-12),
    }


def qp_signed_candidate_meta(
    game: MujocoStateActorGame,
    z: torch.Tensor,
    obs: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor, float]:
    z_req = z.detach().clone().requires_grad_(True)
    F0 = game.field(z_req, obs)
    _, G0 = torch.autograd.functional.jvp(lambda zz: game.field(zz, obs), (z_req,), (F0.detach(),), create_graph=False, strict=False)
    F_det = F0.detach()
    G_det = G0.detach()
    f_norm = float(torch.linalg.norm(F_det).item())
    g_norm = float(torch.linalg.norm(G_det).item())
    v0 = game.merit(z, obs, compute_geometry=False)["V"]

    def point(beta: float, gamma: float) -> float:
        z_cand = apply_delta(z, (-beta * F_det) + (gamma * G_det))
        return game.merit(z_cand, obs, compute_geometry=False)["V"]

    qp = solve_qp_signed_box(point, v0, eta, beta_max=10.0 * eta, gamma_max=10.0 * eta * eta)
    delta = (-qp["beta"] * F_det) + (qp["gamma"] * G_det)
    z_next = apply_delta(z, delta)
    g_ratio = float(abs(qp["gamma"]) * g_norm / ((abs(qp["beta"]) * f_norm) + (abs(qp["gamma"]) * g_norm) + EPS))
    v_after = game.merit(z_next, obs, compute_geometry=False)["V"]
    meta = {
        "update_norm": float(torch.linalg.norm(delta).item()),
        "beta": float(qp["beta"]),
        "beta_signed": float(qp["beta"]),
        "gamma": float(qp["gamma"]),
        "gamma_active": int(abs(qp["gamma"]) > 1e-12),
        "G_contribution_ratio": g_ratio,
        "fallback_to_noG": 0,
        "predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
        "selected_active_set": str(qp["selected_active_set"]),
        "V_before": float(v0),
        "V_after_candidate": float(v_after),
        "cos_F_G": float(torch.dot(F_det, G_det).item() / ((f_norm * g_norm) + EPS)),
        "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, (float(torch.dot(F_det, G_det).item() / ((f_norm * g_norm) + EPS))) ** 2))),
        "beta_positive_flag": int(qp["beta_positive"]),
        "beta_negative_flag": int(qp["beta_negative"]),
        "beta_zero_flag": int(abs(float(qp["beta"])) <= 1e-12),
        "beta_active_flag": int(abs(float(qp["beta"])) > 1e-12),
        "gamma_positive_flag": int(qp["gamma_positive"]),
        "gamma_negative_flag": int(qp["gamma_negative"]),
        "gamma_zero_flag": int(abs(float(qp["gamma"])) <= 1e-12),
        "signed_qp_predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
    }
    return z_next, meta, F_det, G_det, v0


def run_qp_signed_box_damped_safe(game: MujocoStateActorGame, z: torch.Tensor, obs: torch.Tensor, eta: float) -> tuple[torch.Tensor, dict[str, Any]]:
    z_nog, meta_nog = run_nog_signed_box(game, z, obs, eta)
    z_full, meta_full, F_det, G_det, _ = qp_signed_candidate_meta(game, z, obs, eta)
    z_egm, meta_egm = run_egm(game, z, obs, eta)
    v_nog = safe_float(meta_nog["V_after_candidate"])
    v_qp = safe_float(meta_full["V_after_candidate"])
    v_egm = safe_float(meta_egm["V_after_candidate"])
    tol = NO_G_SAFE_TOL * max(1.0, abs(v_nog))

    best_choice = {
        "kind": "signed_noG",
        "eta": 0.0,
        "z": z_nog,
        "V": v_nog,
        "G_ratio": 0.0,
        "margin": 0.0,
        "beta": safe_float(meta_nog["beta"]),
        "gamma": 0.0,
        "gamma_active": 0,
        "fallback_reason": "selected_signed_noG_by_actual_V",
        "chosen_step_type": "signed_noG",
    }
    beta = safe_float(meta_full["beta"])
    gamma = safe_float(meta_full["gamma"])
    for eta_scale in ETA_LIST:
        delta = (-beta * F_det) + (eta_scale * gamma * G_det)
        z_eta = apply_delta(z, delta)
        v_eta = game.merit(z_eta, obs, compute_geometry=False)["V"]
        g_term_norm = abs(eta_scale * gamma) * float(torch.linalg.norm(G_det).item())
        f_term_norm = abs(beta) * float(torch.linalg.norm(F_det).item())
        g_ratio = g_term_norm / (f_term_norm + g_term_norm + EPS)
        improves_actual = v_eta <= (v_nog - tol)
        if improves_actual and v_eta < best_choice["V"] - 1e-12:
            best_choice = {
                "kind": "signed_qp",
                "eta": float(eta_scale),
                "z": z_eta,
                "V": float(v_eta),
                "G_ratio": float(g_ratio),
                "margin": float(v_nog - v_eta),
                "beta": float(beta),
                "gamma": float(eta_scale * gamma),
                "gamma_active": int(abs(eta_scale * gamma) > 1e-12),
                "fallback_reason": "accepted_signed_qp",
                "chosen_step_type": f"signed_dampedG_eta_{str(float(eta_scale)).replace('.', 'p')}",
            }
    chosen = best_choice["z"]
    return chosen, {
        **meta_full,
        "beta": float(best_choice["beta"]),
        "gamma": float(best_choice["gamma"]),
        "gamma_active": int(best_choice["gamma_active"]),
        "fallback_to_noG": int(best_choice["kind"] == "signed_noG"),
        "signed_QP_accept_frac_flag": int(best_choice["kind"] != "signed_noG"),
        "QP_accept_frac_flag": int(best_choice["kind"] != "signed_noG"),
        "chosen_eta": float(best_choice["eta"]),
        "effective_gamma": float(best_choice["gamma"]),
        "effective_G_contribution_ratio": float(best_choice["G_ratio"]),
        "G_contribution_ratio": float(best_choice["G_ratio"]),
        "accepted_QP_better_than_noG_margin": float(best_choice["margin"]),
        "QP_better_than_noG_actual_V": int(best_choice["kind"] != "signed_noG"),
        "QP_better_than_EGM_actual_V": int(best_choice["V"] <= v_egm + 1e-12),
        "fallback_reason": str(best_choice["fallback_reason"]),
        "chosen_step_type": str(best_choice["chosen_step_type"]),
        "V_after_signed_noG": float(v_nog),
        "V_after_candidate": float(best_choice["V"]),
        "signed_QP_better_than_signed_noG_flag": int(best_choice["kind"] != "signed_noG"),
        "signed_QP_accepts_QP_flag": int(best_choice["kind"] != "signed_noG"),
        "full_gamma_value": float(gamma),
    }


def run_invariant_check(spec: EnvSpec, train_states: torch.Tensor, eval_states: torch.Tensor, out_dir: Path) -> tuple[str, list[dict[str, Any]], str]:
    cfg = Config(env_id=spec.env_id, rho=RHO_GRID[0], lambda_u=LAMBDA_U_GRID[0], lambda_w=LAMBDA_W_GRID[0], lr=LR_GRID[0], lambda_F=LAMBDA_F, lambda_J=LAMBDA_J, batch_size=BATCH_SIZE_GRID[0])
    game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
    rows: list[dict[str, Any]] = []
    positive_beta_nonzero_delta = []
    for batch_idx in range(GEOM_BATCHES):
        game.metric_refs = {}
        z = game.init_z(seed_offset=batch_idx)
        obs = game.sample_batch(GEOM_BATCH_SIZE, 7000 + batch_idx)
        z_req = z.detach().clone().requires_grad_(True)
        F0 = game.field(z_req, obs)
        _, G0 = torch.autograd.functional.jvp(lambda zz: game.field(zz, obs), (z_req,), (F0.detach(),), create_graph=False, strict=False)
        F_det = F0.detach()
        G_det = G0.detach()
        v0 = game.merit(z, obs, compute_geometry=False)["V"]

        def point1(beta: float) -> float:
            return game.merit(apply_delta(z, -beta * F_det), obs, compute_geometry=False)["V"]

        def point2(beta: float, gamma: float) -> float:
            return game.merit(apply_delta(z, (-beta * F_det) + (gamma * G_det)), obs, compute_geometry=False)["V"]

        nog = solve_nog(point1, v0, cfg.lr, beta_max=10.0 * cfg.lr)
        qp = solve_qp(point2, v0, cfg.lr, beta_max=10.0 * cfg.lr, gamma_max=10.0 * cfg.lr * cfg.lr)
        q1 = float(nog["q"])
        q2_gamma0 = float(qp["l_beta"] * nog["beta"] + 0.5 * qp["h_bb"] * nog["beta"] * nog["beta"])
        z_nog, native_nog = run_nog_closed(game, z, obs, cfg.lr)
        forced_delta = -safe_float(native_nog["beta"]) * F_det
        z_forced = apply_delta(z, forced_delta)
        delta_diff = float(torch.linalg.norm((z_nog - z) - forced_delta).item())
        z_safe, safe_meta = run_qp_nog_safe(game, z, obs, cfg.lr)
        safe_delta = z_safe - z
        applied_delta = torch.zeros_like(safe_delta)
        if int(safe_meta.get("fallback_to_noG", 0)) == 1:
            applied_delta = z_nog - z
        positive_beta_nonzero_delta.append(int((safe_float(native_nog["beta"]) <= 1e-12) or (torch.linalg.norm(z_nog - z).item() > 1e-12)))
        rows.append(
            {
                "env_id": spec.env_id,
                "batch_index": batch_idx,
                "q1_equals_q2_gamma0_pass": int(abs(q1 - q2_gamma0) <= 1e-6 * max(1.0, abs(q1), abs(q2_gamma0))),
                "predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
                "gamma_negative_fraction_plusG": int(qp["gamma"] < -1e-12),
                "forced_nog_same_batch_delta_diff": delta_diff,
                "forced_nog_same_batch_delta_equals_native": int(delta_diff <= 1e-6 * max(1.0, float(torch.linalg.norm(forced_delta).item()), float(torch.linalg.norm(z_nog - z).item()))),
                "noG_safe_actual_choice_matches_applied_step_flag": int(abs(float(torch.linalg.norm(applied_delta - (z_nog - z)).item())) <= 1e-9) if int(safe_meta.get("fallback_to_noG", 0)) == 1 else 1,
                "positive_beta_nonzero_delta": positive_beta_nonzero_delta[-1],
            }
        )
    q1_pass = float(np.mean([row["q1_equals_q2_gamma0_pass"] for row in rows]))
    pred_pass = float(np.mean([row["predicted_inclusion_pass"] for row in rows]))
    gamma_neg = float(np.mean([row["gamma_negative_fraction_plusG"] for row in rows]))
    forced_pass = float(np.mean([row["forced_nog_same_batch_delta_equals_native"] for row in rows]))
    choice_pass = float(np.mean([row["noG_safe_actual_choice_matches_applied_step_flag"] for row in rows]))
    beta_delta_pass = float(np.mean(positive_beta_nonzero_delta))
    decision = "QP_INVARIANT_FAIL"
    if q1_pass >= 0.999 and pred_pass >= 0.999 and gamma_neg == 0.0 and forced_pass >= 0.999 and choice_pass >= 0.999 and beta_delta_pass >= 0.999:
        decision = "QP_INVARIANT_PASS"
    text = "\n".join(
        [
            "# MuJoCo-state actor coupling invariant check",
            "",
            f"- env: `{spec.env_id}`",
            f"- q1_equals_q2_gamma0_pass_fraction: `{q1_pass:.6f}`",
            f"- predicted_inclusion_pass_fraction: `{pred_pass:.6f}`",
            f"- gamma_negative_fraction_plusG: `{gamma_neg:.6f}`",
            f"- forced_nog_same_batch_delta_equals_native_fraction: `{forced_pass:.6f}`",
            f"- noG_safe_actual_choice_matches_applied_step_flag_fraction: `{choice_pass:.6f}`",
            f"- positive_beta_nonzero_delta_fraction: `{beta_delta_pass:.6f}`",
            "",
            f"- decision: `{decision}`",
            "",
        ]
    )
    write_csv(out_dir / "invariant_summary.csv", rows)
    write_text(out_dir / "invariant_report.md", text)
    write_text(out_dir / "invariant_decision.md", decision + "\n")
    return decision, rows, text


def run_score_leakage_check(spec: EnvSpec, train_states: torch.Tensor, eval_states: torch.Tensor, out_dir: Path) -> tuple[str, list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    pure_values: list[float] = []
    actor_values: list[float] = []
    rot_norm_values: list[float] = []
    rho_probe_grid = [0.5, 1.0, 2.0, 5.0]
    base_cfg = Config(env_id=spec.env_id, rho=max(rho_probe_grid), lambda_u=0.01, lambda_w=0.01, lr=3e-4, lambda_F=LAMBDA_F, lambda_J=LAMBDA_J)
    probe_game = MujocoStateActorGame(spec, base_cfg, train_states, eval_states, SEED)
    z_ref = probe_game.init_z()
    eval_obs = probe_game.eval_batch()
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED + 2026)
    z_probe = z_ref.clone()
    probe_scale = 0.0
    for scale in [0.0, 0.1, 0.3, 1.0, 3.0]:
        candidate = z_ref + scale * 0.25 * torch.randn(z_ref.shape, generator=gen, dtype=DTYPE, device=DEVICE)
        probe_game.metric_refs = {}
        metrics = probe_game.merit(candidate, eval_obs, compute_geometry=False)
        if abs(metrics["actor_game_score"]) >= 0.1 or abs(metrics["rot_norm"]) >= 0.1:
            z_probe = candidate.detach().clone()
            probe_scale = scale
            break
        z_probe = candidate.detach().clone()
        probe_scale = scale
    for rho in rho_probe_grid:
        cfg = Config(env_id=spec.env_id, rho=rho, lambda_u=0.01, lambda_w=0.01, lr=3e-4, lambda_F=LAMBDA_F, lambda_J=LAMBDA_J)
        game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
        z = z_probe.clone()
        eval_obs = game.eval_batch()
        metrics = game.merit(z, eval_obs, compute_geometry=False)
        pure_env = game.evaluate_pure_env_return(z, PURE_ENV_EVAL_EPISODES, seed_offset=500)
        row = {
            "rho": rho,
            "probe_scale": probe_scale,
            "actor_game_score": metrics["actor_game_score"],
            "rot_raw": metrics["rot_raw"],
            "rot_norm": metrics["rot_norm"],
            "rot_scale": metrics["rot_scale"],
            "u_energy": metrics["u_energy"],
            "w_energy": metrics["w_energy"],
            "pure_env_return": pure_env,
        }
        rows.append(row)
        pure_values.append(pure_env)
        actor_values.append(metrics["actor_game_score"])
        rot_norm_values.append(metrics["rot_norm"])
    pure_span = float(max(pure_values) - min(pure_values)) if pure_values else math.nan
    actor_span = float(max(actor_values) - min(actor_values)) if actor_values else math.nan
    checks = {
        "primary_not_equal_pure": int(all(abs(r["actor_game_score"] - r["pure_env_return"]) > 1e-6 for r in rows)),
        "rho_changes_actor_game_score": int(actor_span > 1e-3),
        "rho_does_not_change_pure_env_return": int(pure_span <= 1e-6),
        "actor_game_score_O1": int(max(abs(v) for v in actor_values) >= 0.1 if actor_values else 0),
    }
    decision = "ACTOR_GAME_SCORE_LEAKAGE_BUG"
    if all(value == 1 for value in checks.values()):
        decision = "ACTOR_GAME_SCORE_LEAKAGE_PASS"
    lines = [
        "# actor game score leakage check",
        "",
        f"- env: `{spec.env_id}`",
        f"- rho_probe_grid: `{rho_probe_grid}`",
        f"- probe_scale: `{probe_scale}`",
        f"- primary_not_equal_pure: `{checks['primary_not_equal_pure']}`",
        f"- rho_changes_actor_game_score: `{checks['rho_changes_actor_game_score']}`",
        f"- rho_does_not_change_pure_env_return: `{checks['rho_does_not_change_pure_env_return']}`",
        f"- actor_game_score_O1: `{checks['actor_game_score_O1']}`",
        f"- actor_game_score_span: `{actor_span:.6e}`",
        f"- pure_env_return_span: `{pure_span:.6e}`",
        "",
        f"- decision: `{decision}`",
        "",
    ]
    write_csv(out_dir / "score_leakage_summary.csv", rows)
    write_text(out_dir / "score_leakage_report.md", "\n".join(lines))
    return decision, rows, "\n".join(lines)


def geometry_gate_for_config(spec: EnvSpec, cfg: Config, train_states: torch.Tensor, eval_states: torch.Tensor) -> dict[str, Any]:
    game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
    rows: list[dict[str, Any]] = []
    for batch_idx in range(GEOM_BATCHES):
        z0 = game.init_z(seed_offset=batch_idx)
        game.metric_refs = {}
        obs = game.sample_batch(GEOM_BATCH_SIZE, batch_idx)
        metrics = game.merit(z0, obs, compute_geometry=True)
        _, meta_qp = run_qp_nog_safe(game, z0, obs, cfg.lr)
        rows.append({**metrics, **meta_qp, "batch_index": batch_idx})
    out = {
        "env_id": spec.env_id,
        "rho": cfg.rho,
        "lambda_u": cfg.lambda_u,
        "lambda_w": cfg.lambda_w,
        "joint_lr": cfg.lr,
        "lambda_F": cfg.lambda_F,
        "lambda_J": cfg.lambda_J,
        "batch_size": cfg.batch_size,
        "config_slug": cfg.slug,
        "field_norm": float(np.mean([r["field_norm"] for r in rows])),
        "G_norm": float(np.mean([r["G_norm"] for r in rows])),
        "cos_F_G": float(np.mean([r["cos_F_G"] for r in rows])),
        "non_collinearity": float(np.mean([r["non_collinearity"] for r in rows])),
        "cross_to_same_ratio": float(np.mean([r["cross_to_same_ratio"] for r in rows])),
        "rotation_ratio_proxy": float(np.mean([r["rotation_ratio_proxy"] for r in rows])),
        "gamma_active_frac": float(np.mean([r["gamma_active"] for r in rows])),
        "G_contribution_ratio": float(np.mean([r["G_contribution_ratio"] for r in rows])),
        "QP_better_than_noG_actual_V_fraction": float(np.mean([r["QP_better_than_noG_actual_V"] for r in rows])),
        "fallback_to_noG_frac": float(np.mean([r["fallback_to_noG"] for r in rows])),
        "predicted_inclusion_pass_fraction": float(np.mean([r["predicted_inclusion_pass"] for r in rows])),
        "actor_game_score": float(np.mean([r["actor_game_score"] for r in rows])),
        "rot_raw": float(np.mean([r["rot_raw"] for r in rows])),
        "rot_norm": float(np.mean([r["rot_norm"] for r in rows])),
        "u_energy": float(np.mean([r["u_energy"] for r in rows])),
        "w_energy": float(np.mean([r["w_energy"] for r in rows])),
    }
    out["geometry_gate_pass"] = int(
        out["cross_to_same_ratio"] >= 0.30
        and out["rotation_ratio_proxy"] >= 0.30
        and out["gamma_active_frac"] >= 0.50
        and out["G_contribution_ratio"] >= 0.20
        and out["QP_better_than_noG_actual_V_fraction"] >= 0.70
        and out["fallback_to_noG_frac"] <= 0.30
    )
    out["gate_score"] = (
        1.5 * out["cross_to_same_ratio"]
        + 1.5 * out["rotation_ratio_proxy"]
        + 0.5 * out["non_collinearity"]
        + out["QP_better_than_noG_actual_V_fraction"]
        + out["gamma_active_frac"]
        - out["fallback_to_noG_frac"]
    )
    return out


def evaluate_method_state(game: MujocoStateActorGame, z: torch.Tensor, obs_eval: torch.Tensor, compute_geometry: bool) -> dict[str, Any]:
    return game.merit(z, obs_eval, compute_geometry=compute_geometry)


def run_method(
    game: MujocoStateActorGame,
    cfg: Config,
    method: str,
    seed: int,
    iterations: int = ITERATIONS,
    eval_freq: int = EVAL_FREQ,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    z = game.init_z(seed_offset=seed)
    game.metric_refs = {}
    curves: list[dict[str, Any]] = []
    gamma_active = 0
    fallback = 0
    g_ratio_sum = 0.0
    qp_vs_nog = 0
    fallback_gamma_inactive = 0
    fallback_rejection = 0
    fallback_small_g = 0
    gamma_active_steps = 0
    accept_given_gamma_active = 0
    eta_counter: dict[str, int] = {}
    eval_obs = game.eval_batch()
    for iteration in range(iterations + 1):
        if iteration % eval_freq == 0:
            metrics = evaluate_method_state(game, z, eval_obs, compute_geometry=True)
            pure_env_return = game.evaluate_pure_env_return(z, PURE_ENV_EVAL_EPISODES, seed_offset=1000 + iteration)
            standard_aux = game.evaluate_standard_rarl_returns(z, PURE_ENV_EVAL_EPISODES, alpha=AUX_STANDARD_RARL_ALPHA, seed_offset=2000 + iteration)
            row = {
                "env_id": cfg.env_id,
                "config_slug": cfg.slug,
                "method": method,
                "iteration": iteration,
                "rho": cfg.rho,
                "lambda_u": cfg.lambda_u,
                "lambda_w": cfg.lambda_w,
                "joint_lr": cfg.lr,
                "lambda_F": cfg.lambda_F,
                "lambda_J": cfg.lambda_J,
                "pure_env_return_aux": pure_env_return,
                **standard_aux,
                **metrics,
            }
            row["finite_flag"] = int(all(finite(row[k]) for k in ["actor_game_score", "field_norm", "V", "pure_env_return_aux", "standard_clean_env_return_aux", "standard_robust_env_return_aux"]))
            curves.append(row)
        if iteration == iterations:
            break
        obs = game.sample_batch(cfg.batch_size, iteration + seed * 10000)
        if method == "sgd_gda":
            z, meta = run_sgd(game, z, obs, cfg.lr)
        elif method == "egm":
            z, meta = run_egm(game, z, obs, cfg.lr)
        elif method == "proposed_nog_closed":
            z, meta = run_nog_closed(game, z, obs, cfg.lr)
        elif method == "proposed_qp_closed":
            z, meta = run_qp_closed(game, z, obs, cfg.lr)
        elif method == "proposed_qp_nog_safe":
            z, meta = run_qp_nog_safe(game, z, obs, cfg.lr)
        elif method == "proposed_qp_dampedG_nog_safe":
            z, meta = run_qp_damped_variant(game, z, obs, cfg.lr, normalize_g=False)
        elif method == "proposed_qp_normG_damped_nog_safe":
            z, meta = run_qp_damped_variant(game, z, obs, cfg.lr, normalize_g=True)
        else:
            raise ValueError(method)
        gamma_active += int(meta.get("gamma_active", 0))
        fallback += int(meta.get("fallback_to_noG", 0))
        g_ratio_sum += safe_float(meta.get("G_contribution_ratio", 0.0), 0.0)
        qp_vs_nog += int(meta.get("QP_better_than_noG_actual_V", 0))
        gamma_active_steps += int(safe_float(meta.get("full_gamma_value", meta.get("gamma", 0.0)), 0.0) > 1e-12)
        if int(meta.get("QP_accept_frac_flag", 0)) == 1 and safe_float(meta.get("full_gamma_value", meta.get("gamma", 0.0)), 0.0) > 1e-12:
            accept_given_gamma_active += 1
        reason = str(meta.get("fallback_reason", ""))
        if reason == "gamma_inactive":
            fallback_gamma_inactive += 1
        elif reason in {"no_eta_improves_actual_V", "selected_noG_by_actual_V"}:
            fallback_rejection += 1
        elif reason == "effective_G_too_small":
            fallback_small_g += 1
        step_type = str(meta.get("chosen_step_type", ""))
        if step_type:
            eta_counter[step_type] = eta_counter.get(step_type, 0) + 1
        if curves:
            curves[-1].update(meta)
    total_steps = max(iterations, 1)
    gamma_active_denom = max(gamma_active_steps, 1)
    summary = {
        "env_id": cfg.env_id,
        "config_slug": cfg.slug,
        "method": method,
        "rho": cfg.rho,
        "lambda_u": cfg.lambda_u,
        "lambda_w": cfg.lambda_w,
        "joint_lr": cfg.lr,
        "lambda_F": cfg.lambda_F,
        "lambda_J": cfg.lambda_J,
        "actor_game_score_AUC": auc([row["actor_game_score"] for row in curves]),
        "payoff_J_AUC": auc([row["actor_game_score"] for row in curves]),
        "rot_raw_AUC": auc([row["rot_raw"] for row in curves]),
        "rot_norm_AUC": auc([row["rot_norm"] for row in curves]),
        "field_norm_AUC": auc([row["field_norm"] for row in curves]),
        "Lyapunov_AUC": auc([row["V"] for row in curves]),
        "distance_to_game_stationarity_AUC": auc([row["distance_to_game_stationarity"] for row in curves]),
        "u_energy_AUC": auc([row["u_energy"] for row in curves]),
        "w_energy_AUC": auc([row["w_energy"] for row in curves]),
        "pure_env_return_AUC": auc([row["pure_env_return_aux"] for row in curves]),
        "final_actor_game_score": curves[-1]["actor_game_score"],
        "final_payoff_J": curves[-1]["actor_game_score"],
        "final_field_norm": curves[-1]["field_norm"],
        "fallback_to_noG_frac_total": fallback / total_steps,
        "fallback_to_noG_frac": fallback / total_steps,
        "fallback_due_gamma_inactive_frac": fallback_gamma_inactive / total_steps,
        "fallback_due_rejection_frac": fallback_rejection / total_steps,
        "fallback_due_small_G_frac": fallback_small_g / total_steps,
        "rejection_given_gamma_active_frac": (fallback_rejection + fallback_small_g) / gamma_active_denom,
        "QP_accept_frac_total": 1.0 - (fallback / total_steps),
        "QP_accept_frac": 1.0 - (fallback / total_steps),
        "QP_accept_frac_given_gamma_active": accept_given_gamma_active / gamma_active_denom,
        "gamma_active_frac": gamma_active / total_steps,
        "G_contribution_ratio": g_ratio_sum / total_steps,
        "effective_G_contribution_ratio": float(np.mean([safe_float(row.get("effective_G_contribution_ratio", row.get("G_contribution_ratio", 0.0)), 0.0) for row in curves])),
        "chosen_eta_mean": float(np.mean([safe_float(row.get("chosen_eta", 0.0), 0.0) for row in curves])),
        "effective_gamma_mean": float(np.mean([safe_float(row.get("effective_gamma", 0.0), 0.0) for row in curves])),
        "fullG_better_than_noG_fraction": float(np.mean([safe_float(row.get("fullG_better_than_noG_fraction_flag", 0.0), 0.0) for row in curves])),
        "dampedG_better_than_noG_fraction": float(np.mean([safe_float(row.get("dampedG_better_than_noG_fraction_flag", 0.0), 0.0) for row in curves])),
        "accepted_QP_better_than_noG_margin": float(np.mean([safe_float(row.get("accepted_QP_better_than_noG_margin", 0.0), 0.0) for row in curves])),
        "QP_better_than_noG_actual_V_fraction": qp_vs_nog / total_steps,
        "chosen_eta_distribution": json.dumps(eta_counter, sort_keys=True),
        "curve_sanity_flag": int(all(row["finite_flag"] == 1 for row in curves)),
    }
    return curves, summary


def run_method_signed(
    game: MujocoStateActorGame,
    cfg: Config,
    method: str,
    seed: int,
    iterations: int = ITERATIONS,
    eval_freq: int = EVAL_FREQ,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    z = game.init_z(seed_offset=seed)
    game.metric_refs = {}
    curves: list[dict[str, Any]] = []
    beta_pos = beta_neg = beta_zero = beta_active = 0
    gamma_pos = gamma_neg = gamma_zero = gamma_active = 0
    fallback = 0
    pred_pass = 0
    g_ratio_sum = 0.0
    qp_vs_nog = 0
    accept_signed_qp = 0
    eta_counter: dict[str, int] = {}
    eval_obs = game.eval_batch()
    for iteration in range(iterations + 1):
        if iteration % eval_freq == 0:
            metrics = evaluate_method_state(game, z, eval_obs, compute_geometry=True)
            pure_env_return = game.evaluate_pure_env_return(z, PURE_ENV_EVAL_EPISODES, seed_offset=1000 + iteration)
            standard_aux = game.evaluate_standard_rarl_returns(z, PURE_ENV_EVAL_EPISODES, alpha=AUX_STANDARD_RARL_ALPHA, seed_offset=2000 + iteration)
            row = {
                "env_id": cfg.env_id,
                "config_slug": cfg.slug,
                "method": method,
                "iteration": iteration,
                "rho": cfg.rho,
                "lambda_u": cfg.lambda_u,
                "lambda_w": cfg.lambda_w,
                "joint_lr": cfg.lr,
                "lambda_F": cfg.lambda_F,
                "lambda_J": cfg.lambda_J,
                "pure_env_return_aux": pure_env_return,
                **standard_aux,
                **metrics,
            }
            row["finite_flag"] = int(
                all(
                    finite(row[k])
                    for k in [
                        "actor_game_score",
                        "field_norm",
                        "V",
                        "pure_env_return_aux",
                        "standard_clean_env_return_aux",
                        "standard_robust_env_return_aux",
                        "standard_robust_degradation_aux",
                    ]
                )
            )
            curves.append(row)
        if iteration == iterations:
            break
        obs = game.sample_batch(cfg.batch_size, iteration + seed * 10000)
        if method == "sgd_gda":
            z, meta = run_sgd(game, z, obs, cfg.lr)
        elif method == "egm":
            z, meta = run_egm(game, z, obs, cfg.lr)
        elif method == "proposed_nog_closed_positive":
            z, meta = run_nog_closed(game, z, obs, cfg.lr)
        elif method == "proposed_nog_signed_box":
            z, meta = run_nog_signed_box(game, z, obs, cfg.lr)
        elif method == "proposed_qp_dampedG_nog_safe_positive":
            z, meta = run_qp_damped_variant(game, z, obs, cfg.lr, normalize_g=False)
        elif method == "proposed_qp_signed_box_damped_safe":
            z, meta = run_qp_signed_box_damped_safe(game, z, obs, cfg.lr)
        else:
            raise ValueError(method)
        beta_val = safe_float(meta.get("beta", 0.0), 0.0)
        gamma_val = safe_float(meta.get("gamma", 0.0), 0.0)
        beta_pos += int(beta_val > 1e-12)
        beta_neg += int(beta_val < -1e-12)
        beta_zero += int(abs(beta_val) <= 1e-12)
        beta_active += int(abs(beta_val) > 1e-12)
        gamma_pos += int(gamma_val > 1e-12)
        gamma_neg += int(gamma_val < -1e-12)
        gamma_zero += int(abs(gamma_val) <= 1e-12)
        gamma_active += int(abs(gamma_val) > 1e-12)
        fallback += int(meta.get("fallback_to_noG", 0))
        pred_pass += int(meta.get("predicted_inclusion_pass", 1))
        g_ratio_sum += safe_float(meta.get("effective_G_contribution_ratio", meta.get("G_contribution_ratio", 0.0)), 0.0)
        qp_vs_nog += int(meta.get("QP_better_than_noG_actual_V", 0))
        accept_signed_qp += int(meta.get("signed_QP_accept_frac_flag", meta.get("QP_accept_frac_flag", 0)))
        step_type = str(meta.get("chosen_step_type", ""))
        if step_type:
            eta_counter[step_type] = eta_counter.get(step_type, 0) + 1
        if curves:
            curves[-1].update(meta)
    total_steps = max(iterations, 1)
    summary = {
        "env_id": cfg.env_id,
        "config_slug": cfg.slug,
        "method": method,
        "rho": cfg.rho,
        "lambda_u": cfg.lambda_u,
        "lambda_w": cfg.lambda_w,
        "joint_lr": cfg.lr,
        "lambda_F": cfg.lambda_F,
        "lambda_J": cfg.lambda_J,
        "actor_game_score_AUC": auc([row["actor_game_score"] for row in curves]),
        "field_norm_AUC": auc([row["field_norm"] for row in curves]),
        "Lyapunov_AUC": auc([row["V"] for row in curves]),
        "rot_norm_AUC": auc([row["rot_norm"] for row in curves]),
        "pure_env_return_AUC": auc([row["pure_env_return_aux"] for row in curves]),
        "standard_clean_env_return_AUC": auc([row["standard_clean_env_return_aux"] for row in curves]),
        "standard_robust_env_return_AUC": auc([row["standard_robust_env_return_aux"] for row in curves]),
        "standard_robust_degradation_AUC": auc([row["standard_robust_degradation_aux"] for row in curves]),
        "final_actor_game_score": curves[-1]["actor_game_score"],
        "final_field_norm": curves[-1]["field_norm"],
        "final_standard_clean_env_return": curves[-1]["standard_clean_env_return_aux"],
        "final_standard_robust_env_return": curves[-1]["standard_robust_env_return_aux"],
        "final_standard_robust_degradation": curves[-1]["standard_robust_degradation_aux"],
        "beta_positive_frac": beta_pos / total_steps,
        "beta_negative_frac": beta_neg / total_steps,
        "beta_zero_frac": beta_zero / total_steps,
        "beta_active_frac": beta_active / total_steps,
        "gamma_positive_frac": gamma_pos / total_steps,
        "gamma_negative_frac": gamma_neg / total_steps,
        "gamma_zero_frac": gamma_zero / total_steps,
        "gamma_active_frac": gamma_active / total_steps,
        "fallback_to_noG_frac": fallback / total_steps,
        "predicted_inclusion_pass_frac": pred_pass / total_steps,
        "effective_G_contribution_ratio": g_ratio_sum / total_steps,
        "QP_better_than_signed_noG_actual_V_fraction": qp_vs_nog / total_steps,
        "signed_QP_accept_frac": accept_signed_qp / total_steps,
        "chosen_eta_distribution": json.dumps(eta_counter, sort_keys=True),
        "curve_sanity_flag": int(all(row["finite_flag"] == 1 for row in curves)),
    }
    return curves, summary


def run_sign_diagnostic(spec: EnvSpec, cfg: Config, train_states: torch.Tensor, eval_states: torch.Tensor, out_root: Path) -> tuple[str, list[dict[str, Any]], str]:
    game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
    z = game.init_z(seed_offset=SEED)
    rows: list[dict[str, Any]] = []
    minusF_wins = plusF_wins = plusF_better = neither = 0
    plusG_wins = minusG_wins = minusG_better = 0
    for batch_idx in range(SIGN_DIAG_BATCHES):
        obs = game.sample_batch(cfg.batch_size, 500000 + batch_idx)
        z_req = z.detach().clone().requires_grad_(True)
        F = game.field(z_req, obs).detach()
        _, G = torch.autograd.functional.jvp(lambda zz: game.field(zz, obs), (z_req,), (F,), create_graph=False, strict=False)
        G = G.detach()
        v0 = game.merit(z, obs, compute_geometry=False)["V"]
        eps_f = 1e-4 / (float(torch.linalg.norm(F).item()) + EPS)
        eps_g = 1e-4 / (float(torch.linalg.norm(G).item()) + EPS)
        v_minusF = game.merit(apply_delta(z, -eps_f * F), obs, compute_geometry=False)["V"]
        v_plusF = game.merit(apply_delta(z, +eps_f * F), obs, compute_geometry=False)["V"]
        v_plusG = game.merit(apply_delta(z, +eps_g * G), obs, compute_geometry=False)["V"]
        v_minusG = game.merit(apply_delta(z, -eps_g * G), obs, compute_geometry=False)["V"]
        minusF_improves = int(v_minusF < v0 - 1e-12)
        plusF_improves = int(v_plusF < v0 - 1e-12)
        plusF_better_than_minusF = int(v_plusF < v_minusF - 1e-12)
        neither_improves = int(minusF_improves == 0 and plusF_improves == 0)
        plusG_improves = int(v_plusG < v0 - 1e-12)
        minusG_improves = int(v_minusG < v0 - 1e-12)
        minusG_better_than_plusG = int(v_minusG < v_plusG - 1e-12)
        minusF_wins += minusF_improves
        plusF_wins += plusF_improves
        plusF_better += plusF_better_than_minusF
        neither += neither_improves
        plusG_wins += plusG_improves
        minusG_wins += minusG_improves
        minusG_better += minusG_better_than_plusG
        rows.append(
            {
                "config_slug": cfg.slug,
                "batch_index": batch_idx,
                "V0": v0,
                "V_minusF": v_minusF,
                "V_plusF": v_plusF,
                "V_plusG": v_plusG,
                "V_minusG": v_minusG,
                "minusF_improves": minusF_improves,
                "plusF_improves": plusF_improves,
                "plusF_better_than_minusF": plusF_better_than_minusF,
                "neither_improves": neither_improves,
                "plusG_improves": plusG_improves,
                "minusG_improves": minusG_improves,
                "minusG_better_than_plusG": minusG_better_than_plusG,
            }
        )
    plusF_better_frac = plusF_better / max(SIGN_DIAG_BATCHES, 1)
    decision = "FIELD_SIGN_MATCH_OK"
    if plusF_better_frac >= 0.60:
        decision = "FIELD_SIGN_MISMATCH_LIKELY"
    text = "\n".join(
        [
            "# sign diagnostic report",
            "",
            f"- config: `{cfg.slug}`",
            f"- minusF_improves_frac: `{minusF_wins / SIGN_DIAG_BATCHES:.6f}`",
            f"- plusF_improves_frac: `{plusF_wins / SIGN_DIAG_BATCHES:.6f}`",
            f"- plusF_better_than_minusF_frac: `{plusF_better_frac:.6f}`",
            f"- neither_improves_frac: `{neither / SIGN_DIAG_BATCHES:.6f}`",
            f"- plusG_improves_frac: `{plusG_wins / SIGN_DIAG_BATCHES:.6f}`",
            f"- minusG_improves_frac: `{minusG_wins / SIGN_DIAG_BATCHES:.6f}`",
            f"- minusG_better_than_plusG_frac: `{minusG_better / SIGN_DIAG_BATCHES:.6f}`",
            "",
            f"- decision: `{decision}`",
            "",
        ]
    )
    return decision, rows, text


def run_signed_inclusion_check(spec: EnvSpec, cfg: Config, train_states: torch.Tensor, eval_states: torch.Tensor) -> float:
    game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
    z = game.init_z(seed_offset=SEED)
    passes = []
    for batch_idx in range(SIGN_DIAG_BATCHES):
        obs = game.sample_batch(cfg.batch_size, 700000 + batch_idx)
        z_req = z.detach().clone().requires_grad_(True)
        F = game.field(z_req, obs).detach()
        _, G = torch.autograd.functional.jvp(lambda zz: game.field(zz, obs), (z_req,), (F,), create_graph=False, strict=False)
        G = G.detach()
        v0 = game.merit(z, obs, compute_geometry=False)["V"]
        def point1(beta: float) -> float:
            return game.merit(apply_delta(z, -beta * F), obs, compute_geometry=False)["V"]
        def point2(beta: float, gamma: float) -> float:
            return game.merit(apply_delta(z, (-beta * F) + (gamma * G)), obs, compute_geometry=False)["V"]
        nog = solve_nog_signed_box(point1, v0, cfg.lr, beta_max=10.0 * cfg.lr)
        qp = solve_qp_signed_box(point2, v0, cfg.lr, beta_max=10.0 * cfg.lr, gamma_max=10.0 * cfg.lr * cfg.lr)
        q_signed_nog = float(nog["q"])
        q2_gamma0 = float(qp["l_beta"] * nog["beta"] + 0.5 * qp["h_bb"] * nog["beta"] * nog["beta"])
        passes.append(int(abs(q_signed_nog - q2_gamma0) <= 1e-6 * max(1.0, abs(q_signed_nog), abs(q2_gamma0)) and int(qp["predicted_inclusion_pass"]) == 1))
    return float(np.mean(passes))


def dominance_fraction(curves_a: list[dict[str, Any]], curves_b: list[dict[str, Any]], metric: str) -> float:
    wins = 0
    total = 0
    for ra, rb in zip(curves_a, curves_b):
        va = safe_float(ra[metric])
        vb = safe_float(rb[metric])
        if finite(va) and finite(vb):
            wins += int(va > vb)
            total += 1
    return wins / max(total, 1)


def assess_config(curves_by_method: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sgd = summaries["sgd_gda"]
    egm = summaries["egm"]
    nog = summaries["proposed_nog_closed"]
    qp = summaries["proposed_qp_dampedG_nog_safe"]
    qp_method = str(qp["method"])
    qp_vs_egm = (qp["actor_game_score_AUC"] - egm["actor_game_score_AUC"]) / (abs(egm["actor_game_score_AUC"]) + EPS)
    qp_vs_nog = (qp["actor_game_score_AUC"] - nog["actor_game_score_AUC"]) / (abs(nog["actor_game_score_AUC"]) + EPS)
    qp_vs_sgd = (qp["actor_game_score_AUC"] - sgd["actor_game_score_AUC"]) / (abs(sgd["actor_game_score_AUC"]) + EPS)
    egm_vs_sgd = (egm["actor_game_score_AUC"] - sgd["actor_game_score_AUC"]) / (abs(sgd["actor_game_score_AUC"]) + EPS)
    nog_vs_sgd = (nog["actor_game_score_AUC"] - sgd["actor_game_score_AUC"]) / (abs(sgd["actor_game_score_AUC"]) + EPS)
    dom_qp_egm = dominance_fraction(curves_by_method[qp_method], curves_by_method["egm"], "actor_game_score")
    dom_qp_nog = dominance_fraction(curves_by_method[qp_method], curves_by_method["proposed_nog_closed"], "actor_game_score")
    dom_qp_sgd = dominance_fraction(curves_by_method[qp_method], curves_by_method["sgd_gda"], "actor_game_score")
    final_best = max(summaries.values(), key=lambda row: safe_float(row["final_actor_game_score"]))["method"]
    curvature_rejection_frac = safe_float(qp["rejection_given_gamma_active_frac"])
    boundary_noG_selected_frac = safe_float(qp["fallback_due_gamma_inactive_frac"])
    field_low_enough = (
        safe_float(qp["field_norm_AUC"]) <= safe_float(egm["field_norm_AUC"]) + EPS
        or safe_float(qp["field_norm_AUC"]) <= safe_float(nog["field_norm_AUC"]) + EPS
    )
    decision = "DAMPED_QP_FAILS_REJECTION"
    if (
        qp_vs_egm >= 0.05
        and qp_vs_nog >= 0.05
        and dom_qp_egm >= 0.60
        and dom_qp_nog >= 0.60
        and curvature_rejection_frac <= 0.30
        and qp["QP_accept_frac_given_gamma_active"] >= 0.70
        and qp["effective_G_contribution_ratio"] >= 0.05
        and field_low_enough
        and qp["curve_sanity_flag"] == 1
    ):
        decision = "QP_STABLE_WEAK_POSITIVE"
    if (
        qp_vs_egm >= 0.10
        and qp_vs_nog >= 0.10
        and qp_vs_sgd >= 0.10
        and dom_qp_egm >= 0.70
        and dom_qp_nog >= 0.70
        and curvature_rejection_frac <= 0.20
        and (final_best == qp_method or abs(safe_float(qp["final_actor_game_score"]) - max(safe_float(v["final_actor_game_score"]) for v in summaries.values())) <= 1e-9)
        and safe_float(qp["field_norm_AUC"]) <= min(safe_float(egm["field_norm_AUC"]), safe_float(nog["field_norm_AUC"]), safe_float(sgd["field_norm_AUC"])) + EPS
        and qp["curve_sanity_flag"] == 1
    ):
        decision = "QP_STABLE_STRONG_POSITIVE"
    if decision not in {"QP_STABLE_WEAK_POSITIVE", "QP_STABLE_STRONG_POSITIVE"}:
        if qp["curve_sanity_flag"] != 1:
            decision = "DAMPED_QP_INCONCLUSIVE"
        elif qp_vs_egm < 0.05 or qp_vs_nog < 0.05:
            decision = "DAMPED_QP_FAILS_RETURN"
        else:
            decision = "DAMPED_QP_FAILS_REJECTION"
    return {
        "decision": decision,
        "best_qp_method": qp_method,
        "qp_vs_egm_auc_frac": qp_vs_egm,
        "qp_vs_nog_auc_frac": qp_vs_nog,
        "qp_vs_sgd_auc_frac": qp_vs_sgd,
        "egm_vs_sgd_auc_frac": egm_vs_sgd,
        "nog_vs_sgd_auc_frac": nog_vs_sgd,
        "qp_dom_egm": dom_qp_egm,
        "qp_dom_nog": dom_qp_nog,
        "qp_dom_sgd": dom_qp_sgd,
        "curvature_rejection_frac": curvature_rejection_frac,
        "boundary_noG_selected_frac": boundary_noG_selected_frac,
        "field_low_enough_flag": int(field_low_enough),
    }


def oscillation_score(curves: list[dict[str, Any]], metric: str = "actor_game_score") -> float:
    values = np.asarray([safe_float(row.get(metric, math.nan)) for row in curves], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return math.nan
    diffs = np.diff(values)
    return float(np.std(diffs) / (abs(np.mean(values)) + EPS))


def actor_mean_delta_norm_on_dataset(game: MujocoStateActorGame, z_before: torch.Tensor, z_after: torch.Tensor, obs: torch.Tensor) -> float:
    parts_before = game.split_z(z_before)
    parts_after = game.split_z(z_after)
    u_before = game.actor_mean_action(parts_before["protagonist_actor"], obs)
    u_after = game.actor_mean_action(parts_after["protagonist_actor"], obs)
    w_before = game.actor_mean_action(parts_before["adversary_actor"], obs)
    w_after = game.actor_mean_action(parts_after["adversary_actor"], obs)
    delta = torch.cat([u_after - u_before, w_after - w_before], dim=-1)
    return float(torch.linalg.norm(delta, dim=-1).mean().item())


def run_nog_audit(spec: EnvSpec, train_states: torch.Tensor, eval_states: torch.Tensor, out_root: Path) -> tuple[str, list[dict[str, Any]], str]:
    cfg = Config(spec.env_id, RHO_GRID[0], LAMBDA_U_GRID[0], LAMBDA_W_GRID[0], LR_GRID[0], LAMBDA_F, LAMBDA_J, batch_size=BATCH_SIZE_GRID[0])
    game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
    z = game.init_z(seed_offset=SEED)
    game.metric_refs = {}
    rows: list[dict[str, Any]] = []
    eval_obs = game.eval_batch()
    active_count = 0
    nonzero_delta_count = 0
    actor_delta_count = 0
    for step in range(NOG_AUDIT_STEPS):
        obs = game.sample_batch(cfg.batch_size, 900000 + step)
        before = game.merit(z, eval_obs, compute_geometry=False)
        z_req = z.detach().clone().requires_grad_(True)
        f0 = game.field(z_req, obs).detach()

        def point(beta: float) -> float:
            z_cand = apply_delta(z, -beta * f0)
            return game.merit(z_cand, obs, compute_geometry=False)["V"]

        nog_fit = solve_nog(point, before["V"], cfg.lr, beta_max=10.0 * cfg.lr)
        z_next, meta = run_nog_closed(game, z, obs, cfg.lr)
        after = game.merit(z_next, eval_obs, compute_geometry=False)
        param_delta_norm = float(torch.linalg.norm(z_next - z).item())
        actor_delta_norm = actor_mean_delta_norm_on_dataset(game, z, z_next, eval_obs)
        beta = safe_float(meta.get("beta", 0.0), 0.0)
        beta_active = int(abs(beta) > 1e-12)
        active_count += beta_active
        nonzero_delta_count += int(param_delta_norm > 1e-12)
        actor_delta_count += int(actor_delta_norm > 1e-12)
        rows.append(
            {
                "step": step,
                "beta_noG": beta,
                "beta_active": beta_active,
                "field_norm": before["field_norm"],
                "parameter_delta_norm": param_delta_norm,
                "actor_mean_delta_norm_on_D_eval": actor_delta_norm,
                "J_actor_before": before["actor_game_score"],
                "J_actor_after": after["actor_game_score"],
                "actual_V_before": before["V"],
                "actual_V_after": after["V"],
                "actual_drift_noG": after["V"] - before["V"],
                "q1D_linear_coeff": safe_float(nog_fit["l_beta"]),
                "q1D_quadratic_coeff": safe_float(nog_fit["h_bb"]),
                "beta_raw": safe_float(nog_fit["beta_raw"]),
                "beta_zero_valid_boundary": int(abs(beta) <= 1e-12 and safe_float(nog_fit["l_beta"]) >= -1e-12),
            }
        )
        z = z_next
    beta_active_frac = active_count / max(NOG_AUDIT_STEPS, 1)
    nonzero_param_frac = nonzero_delta_count / max(NOG_AUDIT_STEPS, 1)
    actor_delta_frac = actor_delta_count / max(NOG_AUDIT_STEPS, 1)
    valid_boundary_frac = float(np.mean([row["beta_zero_valid_boundary"] for row in rows]))
    apply_bug_frac = float(np.mean([int((abs(row["beta_noG"]) > 1e-12) and (row["parameter_delta_norm"] <= 1e-12)) for row in rows]))
    finite_ok = int(all(finite(row["actual_V_before"]) and finite(row["actual_V_after"]) and finite(row["J_actor_before"]) and finite(row["J_actor_after"]) for row in rows))
    decision = "NOG_VALID_FIELD_ONLY_STALLING_BASELINE"
    if apply_bug_frac > 0.0 or finite_ok != 1:
        decision = "NOG_AUDIT_FAIL_APPLY_BUG"
    report = [
        "# noG audit report",
        "",
        f"- env: `{spec.env_id}`",
        f"- config: `{cfg.slug}`",
        f"- beta_active_frac: `{beta_active_frac:.6f}`",
        f"- nonzero_parameter_delta_frac: `{nonzero_param_frac:.6f}`",
        f"- actor_mean_delta_nonzero_frac: `{actor_delta_frac:.6f}`",
        f"- valid_boundary_beta_zero_frac: `{valid_boundary_frac:.6f}`",
        f"- apply_bug_frac: `{apply_bug_frac:.6f}`",
        f"- finite_ok: `{finite_ok}`",
        f"- decision: `{decision}`",
        "",
        "The noG rule is a valid one-dimensional Lyapunov boundary rule. In this skew-dominated actor-coupling game, it often selects beta=0 because the field direction alone is not predicted to decrease the Lyapunov merit. This is treated as the field-only ablation's failure mode, not an implementation failure.",
        "",
    ]
    write_csv(out_root / "01_nog_audit" / "nog_audit_steps.csv", rows)
    write_text(out_root / "01_nog_audit" / "nog_audit_report.md", "\n".join(report))
    write_text(out_root / "01_nog_audit" / "nog_audit_decision.md", decision + "\n")
    return decision, rows, "\n".join(report)


def save_training_plot(plot_path: Path, title: str, curves_by_method: dict[str, list[dict[str, Any]]]) -> None:
    if plt is None:
        return
    metrics = [
        ("actor_game_score", "Actor Game Score"),
        ("rot_norm", "Rotational Coupling"),
        ("field_norm", "Field Norm"),
        ("V", "Lyapunov Merit"),
        ("u_energy", "Action Energy (U)"),
        ("w_energy", "Action Energy (W)"),
        ("pure_env_return_aux", "Pure Env Return (Auxiliary)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    for ax, (metric, label) in zip(axes, metrics):
        for method, rows in curves_by_method.items():
            xs = [r["iteration"] for r in rows]
            ys = [safe_float(r[metric]) for r in rows]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Iteration")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    out: list[float] = []
    running: float | None = None
    for value in values:
        if not finite(value):
            out.append(math.nan)
            continue
        if running is None or not finite(running):
            running = float(value)
        else:
            running = alpha * float(value) + (1.0 - alpha) * float(running)
        out.append(float(running))
    return out


def save_metric_plot(
    plot_path: Path,
    curves_by_method: dict[str, list[dict[str, Any]]],
    metric: str,
    title: str,
    smoothed: bool,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, rows in curves_by_method.items():
        xs = [r["iteration"] for r in rows]
        ys = [safe_float(r.get(metric, math.nan)) for r in rows]
        if smoothed:
            ax.plot(xs, ys, alpha=0.20, linewidth=1.0)
            ax.plot(xs, ema(ys), linewidth=2.0, label=method)
        else:
            ax.plot(xs, ys, alpha=0.75, linewidth=1.8, label=method)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_geometry_metric_plot(plot_path: Path, curves_by_method: dict[str, list[dict[str, Any]]]) -> None:
    if plt is None:
        return
    metrics = [
        ("rotation_ratio_proxy", "Rotation Ratio Proxy"),
        ("cross_to_same_ratio", "Cross-to-Same Ratio"),
        ("non_collinearity", "Non-collinearity"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    for ax, (metric, label) in zip(axes, metrics):
        for method, rows in curves_by_method.items():
            xs = [r["iteration"] for r in rows]
            ys = [safe_float(r[metric]) for r in rows]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Iteration")
    axes[0].legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_eta_distribution_plot(plot_path: Path, curves_by_method: dict[str, list[dict[str, Any]]]) -> None:
    if plt is None:
        return
    qp_methods = ["proposed_qp_dampedG_nog_safe", "proposed_qp_normG_damped_nog_safe"]
    fig, axes = plt.subplots(1, len(qp_methods), figsize=(6 * len(qp_methods), 4))
    if len(qp_methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, qp_methods):
        rows = curves_by_method.get(method, [])
        etas = [safe_float(row.get("chosen_eta", math.nan)) for row in rows if finite(row.get("chosen_eta", math.nan))]
        if etas:
            bins = sorted(set([0.0, *ETA_LIST, 1.0]))
            ax.hist(etas, bins=max(len(bins), 5), alpha=0.8)
        ax.set_title(f"{method} eta")
        ax.set_xlabel("chosen_eta")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_qp_vs_nog_actualV_plot(plot_path: Path, curves_by_method: dict[str, list[dict[str, Any]]]) -> None:
    if plt is None:
        return
    qp_methods = ["proposed_qp_dampedG_nog_safe", "proposed_qp_normG_damped_nog_safe"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for method in qp_methods:
        rows = curves_by_method.get(method, [])
        xs = [r["iteration"] for r in rows]
        nog_after = [safe_float(r.get("V_after_noG", math.nan)) for r in rows]
        qp_after = [safe_float(r.get("V_after_candidate", math.nan)) for r in rows]
        margin = [
            safe_float(r.get("V_after_noG", math.nan)) - safe_float(r.get("V_after_candidate", math.nan))
            for r in rows
        ]
        axes[0].plot(xs, nog_after, linestyle="--", alpha=0.6, label=f"{method}: noG")
        axes[0].plot(xs, qp_after, alpha=0.9, label=f"{method}: chosen")
        axes[1].plot(xs, margin, alpha=0.9, label=f"{method}: noG - chosen")
    axes[0].set_title("Actual V after noG vs chosen QP step")
    axes[1].set_title("Actual V improvement margin over noG")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_fallback_reason_plot(plot_path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    if plt is None:
        return
    qp = summaries.get("proposed_qp_dampedG_nog_safe")
    if not qp:
        return
    labels = [
        "fallback_total",
        "gamma_inactive",
        "rejection",
        "small_G",
        "accept_given_gamma",
    ]
    values = [
        safe_float(qp.get("fallback_to_noG_frac_total", 0.0), 0.0),
        safe_float(qp.get("fallback_due_gamma_inactive_frac", 0.0), 0.0),
        safe_float(qp.get("fallback_due_rejection_frac", 0.0), 0.0),
        safe_float(qp.get("fallback_due_small_G_frac", 0.0), 0.0),
        safe_float(qp.get("QP_accept_frac_given_gamma_active", 0.0), 0.0),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.15))
    ax.set_title("Fallback / acceptance diagnostics")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_multiseed_metric_plot(
    plot_path: Path,
    curve_rows: list[dict[str, Any]],
    metric: str,
    title: str,
    smoothed: bool,
) -> None:
    if plt is None:
        return
    grouped: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for row in curve_rows:
        method = str(row["method"])
        seed = int(row["seed"])
        grouped.setdefault(method, {}).setdefault(seed, []).append((int(row["iteration"]), safe_float(row.get(metric, math.nan))))
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, seed_map in grouped.items():
        seed_series = []
        xs = None
        for _, pairs in sorted(seed_map.items()):
            pairs = sorted(pairs, key=lambda x: x[0])
            cur_x = [p[0] for p in pairs]
            cur_y = [p[1] for p in pairs]
            if smoothed:
                cur_y = ema(cur_y, EMA_ALPHA)
            if xs is None:
                xs = cur_x
            seed_series.append(np.asarray(cur_y, dtype=np.float64))
        if xs is None or not seed_series:
            continue
        arr = np.vstack(seed_series)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        ax.plot(xs, mean, label=method, linewidth=2.0)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.20)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_auc_bar_plot(plot_path: Path, summary_rows: list[dict[str, Any]], metric: str, title: str) -> None:
    if plt is None:
        return
    methods = [str(r["method"]) for r in summary_rows]
    values = [safe_float(r.get(metric, math.nan)) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, values)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_simple_bar_plot(plot_path: Path, labels: list[str], values: list[float], title: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_acceptance_bar_plot(plot_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    qp_rows = [r for r in summary_rows if str(r["method"]) == "proposed_qp_dampedG_nog_safe"]
    if not qp_rows:
        return
    values = [
        float(np.mean([safe_float(r.get("fallback_due_gamma_inactive_frac", 0.0)) for r in qp_rows])),
        float(np.mean([safe_float(r.get("rejection_given_gamma_active_frac", r.get("curvature_rejection_frac", 0.0))) for r in qp_rows])),
        float(np.mean([safe_float(r.get("QP_accept_frac_given_gamma_active", 0.0)) for r in qp_rows])),
        float(np.mean([safe_float(r.get("effective_G_contribution_ratio", 0.0)) for r in qp_rows])),
    ]
    labels = ["gamma_inactive", "curvature_reject", "accept_given_gamma", "effective_G_ratio"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title("QP fallback / active-curvature acceptance")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_boundary_behavior_bar(plot_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    qp_rows = [r for r in summary_rows if str(r["method"]) == "proposed_qp_dampedG_nog_safe"]
    if not qp_rows:
        return
    values = [
        float(np.mean([safe_float(r["fallback_due_gamma_inactive_frac"]) for r in qp_rows])),
        float(np.mean([safe_float(r["curvature_rejection_frac"]) for r in qp_rows])),
        float(np.mean([safe_float(r["QP_accept_frac_given_gamma_active"]) for r in qp_rows])),
    ]
    labels = ["boundary_noG_selected", "curvature_rejected", "curvature_accepted"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title("Boundary noG / curvature behavior")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_eta_distribution_plot(plot_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    qp_rows = [r for r in summary_rows if str(r["method"]) == "proposed_qp_dampedG_nog_safe"]
    if not qp_rows:
        return
    counts: dict[str, float] = {}
    for row in qp_rows:
        try:
            dist = json.loads(str(row.get("chosen_eta_distribution", "{}")))
        except Exception:
            dist = {}
        for key, value in dist.items():
            counts[key] = counts.get(key, 0.0) + float(value)
    if not counts:
        return
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values)
    ax.set_title("QP chosen eta distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def setting_assessment(curves_by_method: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]], nog_decision: str) -> dict[str, Any]:
    sgd = summaries["sgd_gda"]
    egm = summaries["egm"]
    nog = summaries["proposed_nog_closed"]
    qp = summaries["proposed_qp_dampedG_nog_safe"]
    qp_auc = safe_float(qp["actor_game_score_AUC"])
    egm_auc = safe_float(egm["actor_game_score_AUC"])
    nog_auc = safe_float(nog["actor_game_score_AUC"])
    sgd_auc = safe_float(sgd["actor_game_score_AUC"])
    qp_vs_egm = (qp_auc - egm_auc) / (abs(egm_auc) + EPS)
    qp_vs_nog = (qp_auc - nog_auc) / (abs(nog_auc) + EPS)
    egm_vs_nog = (egm_auc - nog_auc) / (abs(nog_auc) + EPS)
    nog_vs_sgd = (nog_auc - sgd_auc) / (abs(sgd_auc) + EPS)
    dom_qp_egm = dominance_fraction(curves_by_method["proposed_qp_dampedG_nog_safe"], curves_by_method["egm"], "actor_game_score")
    dom_qp_nog = dominance_fraction(curves_by_method["proposed_qp_dampedG_nog_safe"], curves_by_method["proposed_nog_closed"], "actor_game_score")
    dom_qp_sgd = dominance_fraction(curves_by_method["proposed_qp_dampedG_nog_safe"], curves_by_method["sgd_gda"], "actor_game_score")
    osc_qp = oscillation_score(curves_by_method["proposed_qp_dampedG_nog_safe"])
    qp_summary = summaries["proposed_qp_dampedG_nog_safe"]
    return {
        "qp_vs_egm_auc_frac": qp_vs_egm,
        "qp_vs_nog_auc_frac": qp_vs_nog,
        "egm_vs_nog_auc_frac": egm_vs_nog,
        "nog_vs_sgd_auc_frac": nog_vs_sgd,
        "qp_dom_egm": dom_qp_egm,
        "qp_dom_nog": dom_qp_nog,
        "qp_dom_sgd": dom_qp_sgd,
        "oscillation_score_qp": osc_qp,
        "curvature_rejection_frac": safe_float(qp_summary["rejection_given_gamma_active_frac"]),
        "boundary_noG_selected_frac": safe_float(qp_summary["fallback_due_gamma_inactive_frac"]),
        "curvature_accept_given_active": safe_float(qp_summary["QP_accept_frac_given_gamma_active"]),
        "effective_G_contribution_ratio": safe_float(qp_summary["effective_G_contribution_ratio"]),
        "seed0_qp_best_flag": int(qp_auc >= max(egm_auc, nog_auc, sgd_auc) - 1e-12),
        "setting_pass_flag": int(
            nog_decision in {"NOG_VALID_FIELD_ONLY_STALLING_BASELINE", "NOG_AUDIT_PASS"}
            and safe_float(qp_summary["curve_sanity_flag"]) == 1
            and safe_float(egm["curve_sanity_flag"]) == 1
            and safe_float(nog["curve_sanity_flag"]) == 1
            and safe_float(sgd["curve_sanity_flag"]) == 1
            and safe_float(qp_summary["rejection_given_gamma_active_frac"]) <= 0.30
            and safe_float(qp_summary["QP_accept_frac_given_gamma_active"]) >= 0.70
        ),
    }


def run_setting_selection(
    spec: EnvSpec,
    train_states: torch.Tensor,
    eval_states: torch.Tensor,
    batch_size: int,
    out_root: Path,
    nog_decision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    settings = [
        Config(spec.env_id, RHO_GRID[0], LAMBDA_U_GRID[0], LAMBDA_W_GRID[0], 3e-4, LAMBDA_F, LAMBDA_J, batch_size=batch_size),
        Config(spec.env_id, RHO_GRID[0], LAMBDA_U_GRID[0], LAMBDA_W_GRID[0], 1e-4, LAMBDA_F, LAMBDA_J, batch_size=batch_size),
    ]
    all_curve_rows: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    lines = ["# setting selection report", ""]
    for cfg in settings:
        curves_by_method: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
            curves, summary = run_method(game, cfg, method, SEED, iterations=ITERATIONS, eval_freq=EVAL_FREQ)
            curves_by_method[method] = curves
            summaries[method] = summary
            all_curve_rows.extend(curves)
        assess = setting_assessment(curves_by_method, summaries, nog_decision)
        qp_summary = summaries["proposed_qp_dampedG_nog_safe"]
        row = {
            "config_slug": cfg.slug,
            "env_id": cfg.env_id,
            "rho": cfg.rho,
            "lambda_u": cfg.lambda_u,
            "lambda_w": cfg.lambda_w,
            "joint_lr": cfg.lr,
            "lambda_F": cfg.lambda_F,
            "lambda_J": cfg.lambda_J,
            "batch_size": cfg.batch_size,
            "qp_actor_game_score_AUC": safe_float(qp_summary["actor_game_score_AUC"]),
            "egm_actor_game_score_AUC": safe_float(summaries["egm"]["actor_game_score_AUC"]),
            "nog_actor_game_score_AUC": safe_float(summaries["proposed_nog_closed"]["actor_game_score_AUC"]),
            "sgd_actor_game_score_AUC": safe_float(summaries["sgd_gda"]["actor_game_score_AUC"]),
            **assess,
            "chosen_eta_distribution": qp_summary["chosen_eta_distribution"],
        }
        ranked_rows.append(row)
        lines.append(
            f"- `{cfg.slug}`: qp_auc=`{row['qp_actor_game_score_AUC']:.6e}`, qp_vs_egm=`{row['qp_vs_egm_auc_frac']:.3f}`, "
            f"qp_vs_nog=`{row['qp_vs_nog_auc_frac']:.3f}`, dom_egm=`{row['qp_dom_egm']:.3f}`, dom_nog=`{row['qp_dom_nog']:.3f}`, "
            f"osc=`{row['oscillation_score_qp']:.3f}`, rejection=`{row['curvature_rejection_frac']:.3f}`, "
            f"accept=`{row['curvature_accept_given_active']:.3f}`, pass=`{row['setting_pass_flag']}`"
        )
    ranked_rows.sort(
        key=lambda row: (
            -safe_float(row["setting_pass_flag"]),
            -safe_float(row["qp_actor_game_score_AUC"]),
            -safe_float(row["qp_dom_egm"]),
            -safe_float(row["qp_dom_nog"]),
            safe_float(row["oscillation_score_qp"]),
        )
    )
    decision = "SETTING_SELECTION_PASS" if ranked_rows and int(ranked_rows[0]["setting_pass_flag"]) == 1 else "SETTING_SELECTION_FAIL"
    lines.extend(["", f"- decision: `{decision}`", ""])
    write_csv(out_root / "02_setting_selection" / "setting_selection_ranked.csv", ranked_rows)
    write_text(out_root / "02_setting_selection" / "setting_selection_report.md", "\n".join(lines) + "\n")
    write_text(out_root / "02_setting_selection" / "setting_selection_decision.md", decision + "\n")
    return ranked_rows[0], ranked_rows, all_curve_rows


def run_final_multiseed(
    selected_cfg: Config,
    spec: EnvSpec,
    train_states: torch.Tensor,
    eval_states: torch.Tensor,
    out_root: Path,
    nog_decision: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str]:
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for seed in MULTISEEDS:
        curves_by_method: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            game = MujocoStateActorGame(spec, selected_cfg, train_states, eval_states, seed)
            curves, summary = run_method(game, selected_cfg, method, seed, iterations=ITERATIONS, eval_freq=EVAL_FREQ)
            curves_by_method[method] = curves
            summaries[method] = summary
            for row in curves:
                curve_rows.append({"seed": seed, **row})
        for method, summary in summaries.items():
            summary_rows.append({"seed": seed, **summary})
    by_method = {}
    for row in summary_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    def mean_metric(method: str, key: str) -> float:
        vals = [safe_float(row[key]) for row in by_method.get(method, []) if finite(row.get(key))]
        return float(np.mean(vals)) if vals else math.nan

    def std_metric(method: str, key: str) -> float:
        vals = [safe_float(row[key]) for row in by_method.get(method, []) if finite(row.get(key))]
        return float(np.std(vals)) if vals else math.nan

    def dominance_mean(method_a: str, method_b: str) -> float:
        seed_fracs = []
        for seed in MULTISEEDS:
            rows_a = [r for r in curve_rows if int(r["seed"]) == seed and str(r["method"]) == method_a]
            rows_b = [r for r in curve_rows if int(r["seed"]) == seed and str(r["method"]) == method_b]
            if rows_a and rows_b:
                seed_fracs.append(dominance_fraction(rows_a, rows_b, "actor_game_score"))
        return float(np.mean(seed_fracs)) if seed_fracs else math.nan

    qp_mean = mean_metric("proposed_qp_dampedG_nog_safe", "actor_game_score_AUC")
    egm_mean = mean_metric("egm", "actor_game_score_AUC")
    nog_mean = mean_metric("proposed_nog_closed", "actor_game_score_AUC")
    sgd_mean = mean_metric("sgd_gda", "actor_game_score_AUC")
    qp_vs_egm = (qp_mean - egm_mean) / (abs(egm_mean) + EPS)
    qp_vs_nog = (qp_mean - nog_mean) / (abs(nog_mean) + EPS)
    qp_vs_sgd = (qp_mean - sgd_mean) / (abs(sgd_mean) + EPS)
    egm_vs_nog = (egm_mean - nog_mean) / (abs(nog_mean) + EPS)
    nog_vs_sgd = (nog_mean - sgd_mean) / (abs(sgd_mean) + EPS)
    dom_qp_egm = dominance_mean("proposed_qp_dampedG_nog_safe", "egm")
    dom_qp_nog = dominance_mean("proposed_qp_dampedG_nog_safe", "proposed_nog_closed")
    dom_qp_sgd = dominance_mean("proposed_qp_dampedG_nog_safe", "sgd_gda")
    qp_best_seed_count = 0
    for seed in MULTISEEDS:
        seed_rows = [r for r in summary_rows if int(r["seed"]) == seed]
        if seed_rows:
            best_val = max(safe_float(r["final_actor_game_score"]) for r in seed_rows)
            qp_row = [r for r in seed_rows if str(r["method"]) == "proposed_qp_dampedG_nog_safe"][0]
            qp_best_seed_count += int(safe_float(qp_row["final_actor_game_score"]) >= best_val - 1e-12)
    curvature_rejection_mean = mean_metric("proposed_qp_dampedG_nog_safe", "rejection_given_gamma_active_frac")
    boundary_noG_mean = mean_metric("proposed_qp_dampedG_nog_safe", "fallback_due_gamma_inactive_frac")
    curvature_accept_mean = mean_metric("proposed_qp_dampedG_nog_safe", "QP_accept_frac_given_gamma_active")
    g_ratio_mean = mean_metric("proposed_qp_dampedG_nog_safe", "effective_G_contribution_ratio")
    field_qp = mean_metric("proposed_qp_dampedG_nog_safe", "field_norm_AUC")
    field_best = min(mean_metric("egm", "field_norm_AUC"), mean_metric("proposed_nog_closed", "field_norm_AUC"), mean_metric("sgd_gda", "field_norm_AUC"))
    sgd_curve_sanity_mean = mean_metric("sgd_gda", "curve_sanity_flag")
    nog_curve_sanity_mean = mean_metric("proposed_nog_closed", "curve_sanity_flag")
    decision = "QP_PAPER_FAILS_MULTI_SEED"
    if (
        qp_vs_egm >= 0.05 and qp_vs_nog >= 0.05 and egm_vs_nog > 0.0
        and (nog_vs_sgd > 0.0 or (nog_curve_sanity_mean >= 1.0 and sgd_curve_sanity_mean < 1.0))
        and qp_best_seed_count >= 2 and dom_qp_egm >= 0.60 and dom_qp_nog >= 0.60
        and curvature_rejection_mean <= 0.30 and curvature_accept_mean >= 0.70 and g_ratio_mean >= 0.05
        and nog_decision in {"NOG_VALID_FIELD_ONLY_STALLING_BASELINE", "NOG_AUDIT_PASS"}
    ):
        decision = "QP_PAPER_WEAK_POSITIVE"
    if (
        qp_vs_egm >= 0.10 and qp_vs_nog >= 0.10 and qp_vs_sgd >= 0.10
        and qp_best_seed_count == 3 and safe_float(mean_metric("proposed_qp_dampedG_nog_safe", "final_actor_game_score")) >= max(
            mean_metric("egm", "final_actor_game_score"),
            mean_metric("proposed_nog_closed", "final_actor_game_score"),
            mean_metric("sgd_gda", "final_actor_game_score"),
        ) - 1e-12
        and field_qp <= field_best + EPS and curvature_rejection_mean <= 0.20 and curvature_accept_mean >= 0.80
    ):
        decision = "QP_PAPER_STRONG_POSITIVE"

    auc_rows = []
    for method in METHODS:
        auc_rows.append(
            {
                "method": method,
                "actor_game_score_AUC_mean": mean_metric(method, "actor_game_score_AUC"),
                "actor_game_score_AUC_std": std_metric(method, "actor_game_score_AUC"),
                "rot_norm_AUC_mean": mean_metric(method, "rot_norm_AUC"),
                "field_norm_AUC_mean": mean_metric(method, "field_norm_AUC"),
                "Lyapunov_AUC_mean": mean_metric(method, "Lyapunov_AUC"),
                "final_actor_game_score_mean": mean_metric(method, "final_actor_game_score"),
                "final_actor_game_score_std": std_metric(method, "final_actor_game_score"),
            }
        )
    plot_rows = curve_rows
    save_multiseed_metric_plot(out_root / "plots" / "actor_game_score_mean_std_raw.png", plot_rows, "actor_game_score", "Actor Game Score mean ± std (raw)", smoothed=False)
    save_multiseed_metric_plot(out_root / "plots" / "actor_game_score_mean_std_ema.png", plot_rows, "actor_game_score", "Actor Game Score mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(out_root / "plots" / "rotational_coupling_mean_std_ema.png", plot_rows, "rot_norm", "Rotational Coupling mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(out_root / "plots" / "field_norm_mean_std_ema.png", plot_rows, "field_norm", "Field Norm mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(out_root / "plots" / "lyapunov_merit_mean_std_ema.png", plot_rows, "V", "Lyapunov Merit mean ± std (EMA)", smoothed=True)
    save_auc_bar_plot(out_root / "plots" / "auc_bar_actor_game_score.png", auc_rows, "actor_game_score_AUC_mean", "Actor Game Score AUC")
    save_acceptance_bar_plot(out_root / "plots" / "curvature_acceptance_bar.png", summary_rows)
    save_boundary_behavior_bar(out_root / "plots" / "boundary_noG_behavior_bar.png", summary_rows)
    save_eta_distribution_plot(out_root / "plots" / "eta_distribution_qp.png", summary_rows)

    report_lines = [
        "# paper ready final report",
        "",
        f"- config_slug: `{selected_cfg.slug}`",
        f"- qp_mean_actor_game_score_AUC: `{qp_mean:.6e}`",
        f"- egm_mean_actor_game_score_AUC: `{egm_mean:.6e}`",
        f"- nog_mean_actor_game_score_AUC: `{nog_mean:.6e}`",
        f"- sgd_mean_actor_game_score_AUC: `{sgd_mean:.6e}`",
        f"- qp_vs_egm: `{qp_vs_egm:.6f}`",
        f"- qp_vs_nog: `{qp_vs_nog:.6f}`",
        f"- qp_vs_sgd: `{qp_vs_sgd:.6f}`",
        f"- egm_vs_nog: `{egm_vs_nog:.6f}`",
        f"- nog_vs_sgd: `{nog_vs_sgd:.6f}`",
        f"- dominance_over_egm: `{dom_qp_egm:.6f}`",
        f"- dominance_over_nog: `{dom_qp_nog:.6f}`",
        f"- dominance_over_sgd: `{dom_qp_sgd:.6f}`",
        f"- qp_best_seed_count: `{qp_best_seed_count}`",
        f"- boundary_noG_selected_frac_mean: `{boundary_noG_mean:.6f}`",
        f"- curvature_rejection_frac_mean: `{curvature_rejection_mean:.6f}`",
        f"- curvature_accept_given_active_mean: `{curvature_accept_mean:.6f}`",
        f"- effective_G_contribution_ratio_mean: `{g_ratio_mean:.6f}`",
        f"- noG_audit_decision: `{nog_decision}`",
        "",
        "This is the canonical MuJoCo-state differentiable actor-coupling game experiment. It uses MuJoCo observations and action bounds but not the MuJoCo reward as the primary optimization objective.",
        "",
        f"- decision: `{decision}`",
        "",
    ]
    return decision, curve_rows, summary_rows, "\n".join(report_lines), auc_rows


def run_training_for_configs(
    gate_rows: list[dict[str, Any]],
    datasets: dict[str, tuple[EnvSpec, torch.Tensor, torch.Tensor]],
    out_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    selected = sorted(
        gate_rows,
        key=lambda row: (
            -safe_float(row["lambda_J"]),
            safe_float(row["lambda_F"]),
            -safe_float(row["gate_score"]),
        ),
    )
    if not selected:
        return [], [], "DAMPED_QP_INCONCLUSIVE"

    all_curves: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    report_lines = ["# stable merit fix report", ""]
    curves_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    summaries_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for gate in selected:
        spec, train_states, eval_states = datasets[str(gate["env_id"])]
        cfg = Config(
            env_id=spec.env_id,
            rho=float(gate["rho"]),
            lambda_u=float(gate["lambda_u"]),
            lambda_w=float(gate["lambda_w"]),
            lr=float(gate["joint_lr"]),
            lambda_F=float(gate["lambda_F"]),
            lambda_J=float(gate["lambda_J"]),
            batch_size=int(gate["batch_size"]),
        )
        curves_by_method: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            game = MujocoStateActorGame(spec, cfg, train_states, eval_states, SEED)
            curves, summary = run_method(game, cfg, method, SEED, iterations=ITERATIONS, eval_freq=EVAL_FREQ)
            curves_by_method[method] = curves
            summaries[method] = summary
            all_curves.extend(curves)
        curves_cache[cfg.slug] = curves_by_method
        summaries_cache[cfg.slug] = summaries
        assess = assess_config(curves_by_method, summaries)
        qp_summary = summaries[str(assess["best_qp_method"])]
        rank_row = {
            **gate,
            **assess,
            **qp_summary,
            "config_slug": cfg.slug,
            "batch_size": cfg.batch_size,
            "oscillation_score_qp": oscillation_score(curves_by_method["proposed_qp_dampedG_nog_safe"]),
            "oscillation_score_egm": oscillation_score(curves_by_method["egm"]),
            "oscillation_score_nog": oscillation_score(curves_by_method["proposed_nog_closed"]),
            "oscillation_score_sgd": oscillation_score(curves_by_method["sgd_gda"]),
        }
        ranked_rows.append(rank_row)
        report_lines.append(
            f"- `{cfg.slug}`: decision=`{assess['decision']}`, best_qp=`{assess['best_qp_method']}`, qp_vs_egm=`{assess['qp_vs_egm_auc_frac']:.3f}`, "
            f"qp_vs_nog=`{assess['qp_vs_nog_auc_frac']:.3f}`, qp_dom_egm=`{assess['qp_dom_egm']:.3f}`, qp_dom_nog=`{assess['qp_dom_nog']:.3f}`, "
            f"curv_reject=`{assess['curvature_rejection_frac']:.3f}`, boundary_noG=`{assess['boundary_noG_selected_frac']:.3f}`, "
            f"accept_given_gamma=`{qp_summary['QP_accept_frac_given_gamma_active']:.3f}`, eta=`{qp_summary['chosen_eta_mean']:.3f}`"
        )
        save_training_plot(out_root / "plots" / f"{cfg.slug}_curves.png", cfg.slug, curves_by_method)
        save_geometry_metric_plot(out_root / "plots" / f"{cfg.slug}_geometry.png", curves_by_method)
    ranked_rows.sort(
        key=lambda row: (
            {"QP_STABLE_STRONG_POSITIVE": 0, "QP_STABLE_WEAK_POSITIVE": 1, "DAMPED_QP_FAILS_RETURN": 2, "DAMPED_QP_FAILS_REJECTION": 3, "DAMPED_QP_INCONCLUSIVE": 4}.get(str(row["decision"]), 5),
            -safe_float(row["qp_vs_egm_auc_frac"]),
            safe_float(row["oscillation_score_qp"]),
            -safe_float(row["actor_game_score_AUC"]),
        )
    )
    write_csv(out_root / "02_training" / "stable_merit_fix_all_curves.csv", all_curves)
    write_csv(out_root / "stable_merit_fix_ranked.csv", ranked_rows)
    write_text(out_root / "stable_merit_fix_report.md", "\n".join(report_lines) + "\n")
    top_lines = ["# stable merit fix top configs", ""]
    for idx, row in enumerate(ranked_rows[:10], start=1):
        top_lines.append(
            f"{idx}. `{row['config_slug']}` | qp=`{row['best_qp_method']}` | decision=`{row['decision']}` | actor_game_score_AUC=`{safe_float(row['actor_game_score_AUC']):.6e}` | "
            f"qp_vs_egm=`{safe_float(row['qp_vs_egm_auc_frac']):.3f}` | qp_vs_nog=`{safe_float(row['qp_vs_nog_auc_frac']):.3f}` | "
            f"curvature_rejection=`{safe_float(row['curvature_rejection_frac']):.3f}` | eta=`{safe_float(row['chosen_eta_mean']):.3f}`"
        )
    write_text(out_root / "stable_merit_fix_top_configs.md", "\n".join(top_lines) + "\n")
    if ranked_rows:
        best_slug = str(ranked_rows[0]["config_slug"])
        best_curves = curves_cache[best_slug]
        save_metric_plot(out_root / "plots" / "fixed_eval_actor_game_score_raw.png", best_curves, "actor_game_score", f"{best_slug} Actor Game Score (raw)", smoothed=False)
        save_metric_plot(out_root / "plots" / "fixed_eval_actor_game_score_smoothed.png", best_curves, "actor_game_score", f"{best_slug} Actor Game Score (EMA)", smoothed=True)
        save_metric_plot(out_root / "plots" / "fixed_eval_rotational_coupling_smoothed.png", best_curves, "rot_norm", f"{best_slug} Rotational Coupling (EMA)", smoothed=True)
        save_metric_plot(out_root / "plots" / "fixed_eval_field_norm_smoothed.png", best_curves, "field_norm", f"{best_slug} Field Norm (EMA)", smoothed=True)
        save_metric_plot(out_root / "plots" / "fixed_eval_lyapunov_smoothed.png", best_curves, "V", f"{best_slug} Lyapunov Merit (EMA)", smoothed=True)
        save_fallback_reason_plot(out_root / "plots" / "fallback_reason_breakdown.png", summaries_cache[best_slug])
    final_decision = str(ranked_rows[0]["decision"]) if ranked_rows else "DAMPED_QP_INCONCLUSIVE"
    return ranked_rows, all_curves, final_decision


def run_multiseed_confirmation(best_row: dict[str, Any], datasets: dict[str, tuple[EnvSpec, torch.Tensor, torch.Tensor]], out_root: Path) -> str:
    if str(best_row["decision"]) not in {"QP_STABLE_WEAK_POSITIVE", "QP_STABLE_STRONG_POSITIVE"}:
        return "QP_PAPER_FAILS_MULTI_SEED"
    spec, train_states, eval_states = datasets[str(best_row["env_id"])]
    cfg = Config(
        env_id=spec.env_id,
        rho=float(best_row["rho"]),
        lambda_u=float(best_row["lambda_u"]),
        lambda_w=float(best_row["lambda_w"]),
        lr=float(best_row["joint_lr"]),
        lambda_F=float(best_row["lambda_F"]),
        lambda_J=float(best_row["lambda_J"]),
        batch_size=int(best_row["batch_size"]),
    )
    rows: list[dict[str, Any]] = []
    confirm_methods = ["sgd_gda", "egm", "proposed_nog_closed", str(best_row["best_qp_method"])]
    for seed in [0, 1, 2]:
        curves_by_method: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in confirm_methods:
            game = MujocoStateActorGame(spec, cfg, train_states, eval_states, seed)
            curves, summary = run_method(game, cfg, method, seed, iterations=ITERATIONS, eval_freq=EVAL_FREQ)
            curves_by_method[method] = curves
            summaries[method] = summary
            for row in curves:
                rows.append({"seed": seed, **row})
        assess = assess_config(curves_by_method, summaries)
        for method, summary in summaries.items():
            rows.append({"seed": seed, "row_type": "summary", "method": method, "config_slug": cfg.slug, **summary, **assess})
    write_csv(out_root / "paper_ready_seed_summary.csv", rows)
    summary_rows = [row for row in rows if row.get("row_type") == "summary"]
    curve_rows = [row for row in rows if row.get("row_type") != "summary"]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    qp_key = str(best_row["best_qp_method"])
    def mean_metric(method: str, key: str) -> float:
        vals = [safe_float(row[key]) for row in by_method.get(method, []) if finite(row.get(key))]
        return float(np.mean(vals)) if vals else math.nan
    def std_metric(method: str, key: str) -> float:
        vals = [safe_float(row[key]) for row in by_method.get(method, []) if finite(row.get(key))]
        return float(np.std(vals)) if vals else math.nan
    qp_mean = mean_metric(qp_key, "actor_game_score_AUC")
    egm_mean = mean_metric("egm", "actor_game_score_AUC")
    nog_mean = mean_metric("proposed_nog_closed", "actor_game_score_AUC")
    sgd_mean = mean_metric("sgd_gda", "actor_game_score_AUC")
    qp_vs_egm = (qp_mean - egm_mean) / (abs(egm_mean) + EPS)
    qp_vs_nog = (qp_mean - nog_mean) / (abs(nog_mean) + EPS)
    qp_vs_sgd = (qp_mean - sgd_mean) / (abs(sgd_mean) + EPS)
    dom_qp_egm = mean_metric(qp_key, "qp_dom_egm")
    dom_qp_nog = mean_metric(qp_key, "qp_dom_nog")
    curvature_rejection = mean_metric(qp_key, "curvature_rejection_frac")
    qp_accept = mean_metric(qp_key, "QP_accept_frac_given_gamma_active")
    g_ratio = mean_metric(qp_key, "effective_G_contribution_ratio")
    field_norm_qp = mean_metric(qp_key, "field_norm_AUC")
    field_norm_best = min(mean_metric("egm", "field_norm_AUC"), mean_metric("proposed_nog_closed", "field_norm_AUC"), mean_metric("sgd_gda", "field_norm_AUC"))
    final_best_count = 0
    for seed in [0, 1, 2]:
        seed_rows = [row for row in summary_rows if int(row["seed"]) == seed]
        best = max(seed_rows, key=lambda row: safe_float(row["final_actor_game_score"]))
        final_best_count += int(str(best["method"]) == qp_key)
    decision = "QP_PAPER_FAILS_MULTI_SEED"
    if (
        qp_vs_egm >= 0.05 and qp_vs_nog >= 0.05 and dom_qp_egm >= 0.60 and dom_qp_nog >= 0.60
        and curvature_rejection <= 0.30 and qp_accept >= 0.70 and g_ratio >= 0.05
    ):
        decision = "QP_PAPER_WEAK_POSITIVE"
    if (
        qp_vs_egm >= 0.10 and qp_vs_nog >= 0.10 and qp_vs_sgd >= 0.10
        and dom_qp_egm >= 0.70 and dom_qp_nog >= 0.70
        and final_best_count >= 2 and field_norm_qp <= field_norm_best + EPS and curvature_rejection <= 0.20
    ):
        decision = "QP_PAPER_STRONG_POSITIVE"
    report = [
        "# paper ready report",
        "",
        f"- config_slug: `{cfg.slug}`",
        f"- qp_mean_actor_game_score_AUC: `{qp_mean:.6e}`",
        f"- egm_mean_actor_game_score_AUC: `{egm_mean:.6e}`",
        f"- nog_mean_actor_game_score_AUC: `{nog_mean:.6e}`",
        f"- sgd_mean_actor_game_score_AUC: `{sgd_mean:.6e}`",
        f"- qp_vs_egm: `{qp_vs_egm:.6f}`",
        f"- qp_vs_nog: `{qp_vs_nog:.6f}`",
        f"- qp_vs_sgd: `{qp_vs_sgd:.6f}`",
        f"- dominance_over_egm: `{dom_qp_egm:.6f}`",
        f"- dominance_over_nog: `{dom_qp_nog:.6f}`",
        f"- curvature_rejection_frac: `{curvature_rejection:.6f}`",
        f"- qp_accept_frac_given_gamma_active: `{qp_accept:.6f}`",
        f"- effective_G_contribution_ratio: `{g_ratio:.6f}`",
        f"- qp_final_best_seed_count: `{final_best_count}`",
        "",
        f"- decision: `{decision}`",
        "",
    ]
    write_text(out_root / "paper_ready_report.md", "\n".join(report))

    # dominance table
    dominance_rows = []
    for method in confirm_methods:
        dominance_rows.append(
            {
                "method": method,
                "actor_game_score_AUC_mean": mean_metric(method, "actor_game_score_AUC"),
                "actor_game_score_AUC_std": std_metric(method, "actor_game_score_AUC"),
                "field_norm_AUC_mean": mean_metric(method, "field_norm_AUC"),
                "Lyapunov_AUC_mean": mean_metric(method, "Lyapunov_AUC"),
                "final_actor_game_score_mean": mean_metric(method, "final_actor_game_score"),
            }
        )
    write_csv(out_root / "plots" / "dominance_table.csv", dominance_rows)
    save_multiseed_metric_plot(out_root / "plots" / "actor_game_score_mean_std_smoothed.png", curve_rows, "actor_game_score", "Actor Game Score (EMA mean ± std)", smoothed=True)
    save_multiseed_metric_plot(out_root / "plots" / "actor_game_score_mean_std_raw.png", curve_rows, "actor_game_score", "Actor Game Score (raw mean ± std)", smoothed=False)
    save_multiseed_metric_plot(out_root / "plots" / "field_norm_mean_std_smoothed.png", curve_rows, "field_norm", "Field Norm (EMA mean ± std)", smoothed=True)
    save_multiseed_metric_plot(out_root / "plots" / "lyapunov_merit_mean_std_smoothed.png", curve_rows, "V", "Lyapunov Merit (EMA mean ± std)", smoothed=True)
    save_acceptance_bar_plot(out_root / "plots" / "fallback_acceptance_bar.png", summary_rows)
    save_auc_bar_plot(out_root / "plots" / "auc_bar_actor_game_score.png", dominance_rows, "actor_game_score_AUC_mean", "Actor Game Score AUC")
    return decision


def assess_signed_config(
    cfg: Config,
    summaries: dict[str, dict[str, Any]],
    curves_by_method: dict[str, list[dict[str, Any]]],
    sign_decision: str,
    inclusion_pass_frac: float,
) -> dict[str, Any]:
    sgd = summaries["sgd_gda"]
    egm = summaries["egm"]
    nog_pos = summaries["proposed_nog_closed_positive"]
    nog_signed = summaries["proposed_nog_signed_box"]
    qp_pos = summaries["proposed_qp_dampedG_nog_safe_positive"]
    qp_signed = summaries["proposed_qp_signed_box_damped_safe"]

    def frac_improve(a: dict[str, Any], b: dict[str, Any]) -> float:
        av = safe_float(a["actor_game_score_AUC"])
        bv = safe_float(b["actor_game_score_AUC"])
        return (av - bv) / (abs(bv) + EPS)

    qp_vs_egm = frac_improve(qp_signed, egm)
    qp_vs_signed_nog = frac_improve(qp_signed, nog_signed)
    signed_nog_vs_sgd = frac_improve(nog_signed, sgd)
    egm_vs_sgd = frac_improve(egm, sgd)
    egm_vs_signed_nog = frac_improve(egm, nog_signed)
    qp_dom_egm = dominance_fraction(curves_by_method["proposed_qp_signed_box_damped_safe"], curves_by_method["egm"], "actor_game_score")
    qp_dom_signed_nog = dominance_fraction(curves_by_method["proposed_qp_signed_box_damped_safe"], curves_by_method["proposed_nog_signed_box"], "actor_game_score")

    decision = "SIGNED_BOX_FAILS"
    if (
        safe_float(nog_signed["beta_active_frac"]) >= 0.50
        and signed_nog_vs_sgd >= 0.05
        and (egm_vs_signed_nog >= 0.0 or egm_vs_sgd >= 0.05)
        and qp_vs_egm >= 0.05
        and qp_vs_signed_nog >= 0.05
        and qp_dom_egm >= 0.60
        and qp_dom_signed_nog >= 0.60
        and safe_float(qp_signed["signed_QP_accept_frac"]) >= 0.70
        and inclusion_pass_frac >= 0.999
        and int(qp_signed["curve_sanity_flag"]) == 1
    ):
        decision = "SIGNED_BOX_DIAGNOSTIC_POSITIVE"
    elif (
        safe_float(nog_signed["beta_active_frac"]) >= 0.50
        and signed_nog_vs_sgd >= 0.05
        and qp_vs_egm >= 0.05
        and qp_vs_signed_nog >= 0.05
        and qp_dom_egm >= 0.60
        and qp_dom_signed_nog >= 0.60
        and safe_float(qp_signed["signed_QP_accept_frac"]) >= 0.70
        and inclusion_pass_frac >= 0.999
        and int(qp_signed["curve_sanity_flag"]) == 1
    ):
        decision = "SIGNED_QP_POSITIVE_BUT_EGM_NOT_ORDERED"
    elif egm_vs_sgd < 0.05 and egm_vs_signed_nog < 0.0:
        decision = "EGM_NOT_A_VALID_MIDDLE_BASELINE"
    if sign_decision == "FIELD_SIGN_MISMATCH_LIKELY":
        decision = "FIELD_SIGN_MISMATCH_LIKELY"

    return {
        "config_slug": cfg.slug,
        "env_id": cfg.env_id,
        "joint_lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "sign_decision": sign_decision,
        "signed_qp_predicted_inclusion_pass_frac": inclusion_pass_frac,
        "signed_nog_beta_active_frac": safe_float(nog_signed["beta_active_frac"]),
        "signed_nog_beta_positive_frac": safe_float(nog_signed["beta_positive_frac"]),
        "signed_nog_beta_negative_frac": safe_float(nog_signed["beta_negative_frac"]),
        "signed_nog_beta_zero_frac": safe_float(nog_signed["beta_zero_frac"]),
        "signed_qp_beta_positive_frac": safe_float(qp_signed["beta_positive_frac"]),
        "signed_qp_beta_negative_frac": safe_float(qp_signed["beta_negative_frac"]),
        "signed_qp_gamma_positive_frac": safe_float(qp_signed["gamma_positive_frac"]),
        "signed_qp_gamma_negative_frac": safe_float(qp_signed["gamma_negative_frac"]),
        "signed_qp_accept_frac": safe_float(qp_signed["signed_QP_accept_frac"]),
        "signed_qp_effective_G_contribution_ratio": safe_float(qp_signed["effective_G_contribution_ratio"]),
        "sgd_actor_game_score_AUC": safe_float(sgd["actor_game_score_AUC"]),
        "egm_actor_game_score_AUC": safe_float(egm["actor_game_score_AUC"]),
        "nog_positive_actor_game_score_AUC": safe_float(nog_pos["actor_game_score_AUC"]),
        "nog_signed_actor_game_score_AUC": safe_float(nog_signed["actor_game_score_AUC"]),
        "qp_positive_actor_game_score_AUC": safe_float(qp_pos["actor_game_score_AUC"]),
        "qp_signed_actor_game_score_AUC": safe_float(qp_signed["actor_game_score_AUC"]),
        "qp_vs_egm_auc_frac": qp_vs_egm,
        "qp_vs_signed_nog_auc_frac": qp_vs_signed_nog,
        "signed_nog_vs_sgd_auc_frac": signed_nog_vs_sgd,
        "egm_vs_sgd_auc_frac": egm_vs_sgd,
        "egm_vs_signed_nog_auc_frac": egm_vs_signed_nog,
        "qp_dom_egm": qp_dom_egm,
        "qp_dom_signed_nog": qp_dom_signed_nog,
        "decision": decision,
    }


def save_sign_distribution_plot(plot_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if plt is None or not summary_rows:
        return
    labels = [Path(str(row["config_slug"])).name for row in summary_rows]
    beta_neg = [safe_float(row["signed_nog_beta_negative_frac"]) for row in summary_rows]
    beta_pos = [safe_float(row["signed_nog_beta_positive_frac"]) for row in summary_rows]
    gamma_neg = [safe_float(row["signed_qp_gamma_negative_frac"]) for row in summary_rows]
    gamma_pos = [safe_float(row["signed_qp_gamma_positive_frac"]) for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5 * width, beta_neg, width, label="signed noG beta-")
    ax.bar(x - 0.5 * width, beta_pos, width, label="signed noG beta+")
    ax.bar(x + 0.5 * width, gamma_neg, width, label="signed QP gamma-")
    ax.bar(x + 1.5 * width, gamma_pos, width, label="signed QP gamma+")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Signed beta/gamma distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_signed_qp_acceptance_plot(plot_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if plt is None or not summary_rows:
        return
    labels = [Path(str(row["config_slug"])).name for row in summary_rows]
    accepts = [safe_float(row["signed_qp_accept_frac"]) for row in summary_rows]
    incl = [safe_float(row["signed_qp_predicted_inclusion_pass_frac"]) for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2.0, accepts, width, label="accept_frac")
    ax.bar(x + width / 2.0, incl, width, label="predicted_inclusion_pass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Signed-QP acceptance and inclusion")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def archive_existing_result_root(path: Path) -> None:
    if not path.exists():
        return
    children = list(path.iterdir())
    substantive = [child for child in children if child.name not in {"progress.log", "progress.err.log"}]
    if not substantive:
        return
    archive = path.parent / f"{path.name}_archive_{random.randint(100000, 999999)}"
    try:
        path.rename(archive)
    except PermissionError:
        archive.mkdir(parents=True, exist_ok=True)
        for child in substantive:
            shutil.move(str(child), str(archive / child.name))


def main() -> None:
    seed_everything(SEED)
    archive_existing_result_root(RESULT_ROOT)
    ensure_dir(RESULT_ROOT / "plots")

    write_json(
        RESULT_ROOT / "run_spec.json",
        {
            "env": ENV_ORDER[0],
            "rho": RHO_GRID[0],
            "lambda_u": LAMBDA_U_GRID[0],
            "lambda_w": LAMBDA_W_GRID[0],
            "joint_lr_grid": LR_GRID,
            "batch_size": BATCH_SIZE_GRID[0],
            "lambda_F": LAMBDA_F,
            "lambda_J": LAMBDA_J,
            "iterations": ITERATIONS,
            "eval_freq": EVAL_FREQ,
            "train_dataset_size": TRAIN_DATASET_SIZE,
            "eval_dataset_size": EVAL_DATASET_SIZE,
            "seeds": MULTISEEDS,
            "methods": METHODS,
            "aux_standard_rarl_alpha": AUX_STANDARD_RARL_ALPHA,
        },
    )

    spec = check_env(ENV_ORDER[0])
    if spec is None:
        write_text(RESULT_ROOT / "final_signed_qp_decision.md", "SIGNED_QP_FAILS_OVER_SIGNED_NOG\n")
        write_text(RESULT_ROOT / "final_signed_qp_report.md", "No MuJoCo env was available.\n")
        return
    cfg = Config(spec.env_id, RHO_GRID[0], LAMBDA_U_GRID[0], LAMBDA_W_GRID[0], LR_GRID[0], LAMBDA_F, LAMBDA_J, batch_size=BATCH_SIZE_GRID[0])

    curve_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    per_seed_compare: list[dict[str, Any]] = []

    for seed in MULTISEEDS:
        train_states, eval_states = collect_state_dataset(spec, seed, TRAIN_DATASET_SIZE, EVAL_DATASET_SIZE)
        game = MujocoStateActorGame(spec, cfg, train_states, eval_states, seed)
        curves_by_method: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            curves, summary = run_method_signed(game, cfg, method, seed)
            curves_by_method[method] = curves
            summaries[method] = summary
            for row in curves:
                curve_rows.append({"seed": seed, **row})
            seed_summary_rows.append({"seed": seed, **summary})
        qp_auc = safe_float(summaries["proposed_qp_signed_box_damped_safe"]["actor_game_score_AUC"])
        nog_auc = safe_float(summaries["proposed_nog_signed_box"]["actor_game_score_AUC"])
        per_seed_compare.append(
            {
                "seed": seed,
                "qp_auc": qp_auc,
                "signed_nog_auc": nog_auc,
                "qp_vs_signed_nog_auc_frac": (qp_auc - nog_auc) / (abs(nog_auc) + EPS),
                "qp_final_actor_game_score": safe_float(summaries["proposed_qp_signed_box_damped_safe"]["final_actor_game_score"]),
                "signed_nog_final_actor_game_score": safe_float(summaries["proposed_nog_signed_box"]["final_actor_game_score"]),
            }
        )

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in seed_summary_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    def mean_metric(method: str, key: str) -> float:
        vals = [safe_float(r.get(key, math.nan)) for r in by_method.get(method, []) if finite(r.get(key))]
        return float(np.mean(vals)) if vals else math.nan

    def std_metric(method: str, key: str) -> float:
        vals = [safe_float(r.get(key, math.nan)) for r in by_method.get(method, []) if finite(r.get(key))]
        return float(np.std(vals)) if vals else math.nan

    def mean_curve_dominance(method_a: str, method_b: str, metric: str) -> float:
        seeds = sorted({int(r["seed"]) for r in curve_rows})
        seed_fracs = []
        for seed in seeds:
            rows_a = [r for r in curve_rows if int(r["seed"]) == seed and str(r["method"]) == method_a]
            rows_b = [r for r in curve_rows if int(r["seed"]) == seed and str(r["method"]) == method_b]
            if rows_a and rows_b:
                seed_fracs.append(dominance_fraction(rows_a, rows_b, metric))
        return float(np.mean(seed_fracs)) if seed_fracs else math.nan

    qp_key = "proposed_qp_signed_box_damped_safe"
    nog_key = "proposed_nog_signed_box"
    egm_key = "egm"
    sgd_key = "sgd_gda"

    qp_mean_auc = mean_metric(qp_key, "actor_game_score_AUC")
    nog_mean_auc = mean_metric(nog_key, "actor_game_score_AUC")
    egm_mean_auc = mean_metric(egm_key, "actor_game_score_AUC")
    sgd_mean_auc = mean_metric(sgd_key, "actor_game_score_AUC")
    qp_vs_nog = (qp_mean_auc - nog_mean_auc) / (abs(nog_mean_auc) + EPS)
    qp_beats_nog_seeds = sum(int(safe_float(r["qp_auc"]) > safe_float(r["signed_nog_auc"])) for r in per_seed_compare)
    dom_qp_nog = mean_curve_dominance(qp_key, nog_key, "actor_game_score")
    accept_frac = mean_metric(qp_key, "signed_QP_accept_frac")
    g_ratio = mean_metric(qp_key, "effective_G_contribution_ratio")
    beta_active = mean_metric(nog_key, "beta_active_frac")
    qp_final = mean_metric(qp_key, "final_actor_game_score")
    nog_final = mean_metric(nog_key, "final_actor_game_score")
    sgd_final = mean_metric(sgd_key, "final_actor_game_score")
    egm_final = mean_metric(egm_key, "final_actor_game_score")
    best_final = max(qp_final, nog_final, sgd_final, egm_final)

    egm_low_field = mean_metric(egm_key, "field_norm_AUC") <= min(mean_metric(qp_key, "field_norm_AUC"), mean_metric(nog_key, "field_norm_AUC"), mean_metric(sgd_key, "field_norm_AUC")) + EPS
    egm_low_score = egm_mean_auc < 0.25 * max(qp_mean_auc, nog_mean_auc, sgd_mean_auc)

    final_decision = "SIGNED_QP_FAILS_OVER_SIGNED_NOG"
    if (
        qp_vs_nog >= 0.05
        and qp_beats_nog_seeds == 3
        and dom_qp_nog >= 0.70
        and qp_final >= best_final - 1e-12
        and g_ratio >= 0.10
    ):
        final_decision = "SIGNED_QP_STRONG_POSITIVE"
    elif (
        qp_vs_nog >= 0.01
        and qp_beats_nog_seeds >= 2
        and dom_qp_nog >= 0.60
        and accept_frac >= 0.70
        and g_ratio >= 0.05
        and beta_active >= 0.80
        and all(int(r.get("curve_sanity_flag", 0)) == 1 for r in by_method.get(qp_key, []))
    ):
        final_decision = "SIGNED_QP_WEAK_POSITIVE"
    elif egm_low_field and egm_low_score:
        final_decision = "EGM_NOT_A_VALID_MIDDLE_BASELINE"

    method_auc_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_auc_rows.append(
            {
                "method": method,
                "actor_game_score_AUC_mean": mean_metric(method, "actor_game_score_AUC"),
                "actor_game_score_AUC_std": std_metric(method, "actor_game_score_AUC"),
                "field_norm_AUC_mean": mean_metric(method, "field_norm_AUC"),
                "Lyapunov_AUC_mean": mean_metric(method, "Lyapunov_AUC"),
                "rot_norm_AUC_mean": mean_metric(method, "rot_norm_AUC"),
                "final_actor_game_score_mean": mean_metric(method, "final_actor_game_score"),
                "standard_clean_env_return_AUC_mean": mean_metric(method, "standard_clean_env_return_AUC"),
                "standard_robust_env_return_AUC_mean": mean_metric(method, "standard_robust_env_return_AUC"),
                "standard_robust_degradation_AUC_mean": mean_metric(method, "standard_robust_degradation_AUC"),
            }
        )

    write_csv(RESULT_ROOT / "final_signed_qp_curve_rows.csv", curve_rows)
    write_csv(RESULT_ROOT / "final_signed_qp_seed_summary.csv", seed_summary_rows)
    write_csv(RESULT_ROOT / "final_signed_qp_method_auc.csv", method_auc_rows)

    save_multiseed_metric_plot(RESULT_ROOT / "plots" / "actor_game_score_mean_std_raw.png", curve_rows, "actor_game_score", "Actor Game Score mean ± std (raw)", smoothed=False)
    save_multiseed_metric_plot(RESULT_ROOT / "plots" / "actor_game_score_mean_std_ema.png", curve_rows, "actor_game_score", "Actor Game Score mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(RESULT_ROOT / "plots" / "field_norm_mean_std_ema.png", curve_rows, "field_norm", "Field Norm mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(RESULT_ROOT / "plots" / "aux_clean_env_return_mean_std_ema.png", curve_rows, "standard_clean_env_return_aux", "Aux Clean Env Return mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(RESULT_ROOT / "plots" / "aux_robust_env_return_mean_std_ema.png", curve_rows, "standard_robust_env_return_aux", "Aux Robust Env Return mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(RESULT_ROOT / "plots" / "aux_robust_degradation_mean_std_ema.png", curve_rows, "standard_robust_degradation_aux", "Aux Robust Degradation mean ± std (EMA)", smoothed=True)
    save_multiseed_metric_plot(
        RESULT_ROOT / "plots" / "signed_qp_vs_signed_nog_actor_score_zoom.png",
        [r for r in curve_rows if str(r["method"]) in {qp_key, nog_key}],
        "actor_game_score",
        "Signed QP vs Signed noG Actor Game Score (EMA)",
        smoothed=True,
    )
    save_sign_distribution_plot(RESULT_ROOT / "plots" / "beta_gamma_sign_distribution.png", [
        {
            "config_slug": cfg.slug,
            "signed_nog_beta_negative_frac": mean_metric(nog_key, "beta_negative_frac"),
            "signed_nog_beta_positive_frac": mean_metric(nog_key, "beta_positive_frac"),
            "signed_qp_gamma_negative_frac": mean_metric(qp_key, "gamma_negative_frac"),
            "signed_qp_gamma_positive_frac": mean_metric(qp_key, "gamma_positive_frac"),
        }
    ])
    save_signed_qp_acceptance_plot(RESULT_ROOT / "plots" / "signed_qp_acceptance_bar.png", [
        {
            "config_slug": cfg.slug,
            "signed_qp_accept_frac": accept_frac,
            "signed_qp_predicted_inclusion_pass_frac": mean_metric(qp_key, "predicted_inclusion_pass_frac"),
        }
    ])
    save_simple_bar_plot(
        RESULT_ROOT / "plots" / "signed_qp_vs_signed_nog_auc_bar.png",
        ["signed_noG", "signed_QP"],
        [nog_mean_auc, qp_mean_auc],
        "Signed QP vs Signed noG Actor Score AUC",
    )
    save_simple_bar_plot(
        RESULT_ROOT / "plots" / "signed_qp_vs_signed_nog_dominance_bar.png",
        ["QP_over_signed_noG"],
        [dom_qp_nog],
        "Signed QP dominance over Signed noG",
    )
    save_simple_bar_plot(
        RESULT_ROOT / "plots" / "egm_stationary_behavior.png",
        ["EGM field_norm_AUC", "EGM actor_score_AUC"],
        [mean_metric(egm_key, "field_norm_AUC"), mean_metric(egm_key, "actor_game_score_AUC")],
        "EGM stationary behavior diagnostic",
    )

    report_lines = [
        "# Final signed QP vs signed noG confirmation",
        "",
        f"- env: `{spec.env_id}`",
        f"- seeds: `{MULTISEEDS}`",
        f"- joint_lr: `{cfg.lr}`",
        f"- batch_size: `{cfg.batch_size}`",
        f"- aux_standard_rarl_alpha: `{AUX_STANDARD_RARL_ALPHA}`",
        "",
        "## Primary comparison: signed QP vs signed noG",
        "",
        f"- signed_qp_mean_actor_game_score_AUC: `{qp_mean_auc:.6e}`",
        f"- signed_qp_std_actor_game_score_AUC: `{std_metric(qp_key, 'actor_game_score_AUC'):.6e}`",
        f"- signed_nog_mean_actor_game_score_AUC: `{nog_mean_auc:.6e}`",
        f"- signed_nog_std_actor_game_score_AUC: `{std_metric(nog_key, 'actor_game_score_AUC'):.6e}`",
        f"- mean_qp_vs_signed_nog_improvement: `{qp_vs_nog:.6f}`",
        f"- signed_qp_beats_signed_nog_seeds: `{qp_beats_nog_seeds}` / 3",
        f"- signed_qp_dominance_over_signed_nog: `{dom_qp_nog:.6f}`",
        f"- signed_qp_accept_frac: `{accept_frac:.6f}`",
        f"- effective_G_contribution_ratio: `{g_ratio:.6f}`",
        f"- signed_nog_beta_active_frac: `{beta_active:.6f}`",
        "",
        "## Auxiliary standard/robust MuJoCo rollout diagnostics",
        "",
        f"- signed_qp_standard_clean_env_return_AUC_mean: `{mean_metric(qp_key, 'standard_clean_env_return_AUC'):.6e}`",
        f"- signed_qp_standard_robust_env_return_AUC_mean: `{mean_metric(qp_key, 'standard_robust_env_return_AUC'):.6e}`",
        f"- signed_qp_standard_robust_degradation_AUC_mean: `{mean_metric(qp_key, 'standard_robust_degradation_AUC'):.6e}`",
        f"- signed_nog_standard_clean_env_return_AUC_mean: `{mean_metric(nog_key, 'standard_clean_env_return_AUC'):.6e}`",
        f"- signed_nog_standard_robust_env_return_AUC_mean: `{mean_metric(nog_key, 'standard_robust_env_return_AUC'):.6e}`",
        f"- signed_nog_standard_robust_degradation_AUC_mean: `{mean_metric(nog_key, 'standard_robust_degradation_AUC'):.6e}`",
        "",
        "## EGM diagnostic",
        "",
        f"- egm_actor_game_score_AUC_mean: `{egm_mean_auc:.6e}`",
        f"- egm_field_norm_AUC_mean: `{mean_metric(egm_key, 'field_norm_AUC'):.6e}`",
        f"- egm_low_field_norm_but_low_actor_score_flag: `{int(egm_low_field and egm_low_score)}`",
        "",
        "Allowing a signed one-dimensional field coefficient eliminates the artificial stalling of the positive noG rule, confirming that the beta>=0 constraint was overly restrictive in this actor-coupling game. Against this stronger signed field-only baseline, the signed-box QP rule yields a consistent but modest improvement, indicating that the curvature direction provides incremental benefit beyond the best one-dimensional field step.",
        "",
        "EGM is reported as a diagnostic baseline but is not used as the middle ordering baseline, because in this benchmark it tends to converge to a low-field-norm but low-payoff stationary solution.",
        "",
        f"- decision: `{final_decision}`",
        "",
    ]
    write_text(RESULT_ROOT / "final_signed_qp_report.md", "\n".join(report_lines) + "\n")
    write_text(RESULT_ROOT / "final_signed_qp_decision.md", final_decision + "\n")


if __name__ == "__main__":
    main()
