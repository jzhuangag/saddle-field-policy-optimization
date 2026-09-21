"""RARL-v3 subsection 3: geometry-screened neural MuJoCo RARL wrappers.

This script is deliberately isolated from the older RARL/QP attempts.  It does not
modify the reward, does not inject a direct u^T B v payoff term, and writes all
artifacts under results/rarl_v3_geometry_screened_mujoco.

Pipeline:
  1. Build adversarial actuator wrappers a_env = clip(u + sigma * B v).
  2. Train a short frozen smooth joint critic Q(s,u,v) on original env reward.
  3. Measure neural-parameter saddle-field geometry.
  4. Run a compact method comparison only for geometry-promising settings.

The run is a compute-budgeted screen, not a universal MuJoCo benchmark claim.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORK_ROOT = Path(r"C:\Users\jzhuangag\work\rarl")
os.environ.setdefault("CONDA_PREFIX", str(WORK_ROOT / "py310-runtime"))
sys.path.insert(0, str(WORK_ROOT))
sys.path.insert(0, str(WORK_ROOT / "original" / "torch-RARL" / "scripts"))

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import semi_joint_ptau_standard_rarl_qp as M  # noqa: E402
import phaseO_pathwise_critic as P  # noqa: E402
from phaseO_rotcheck import matrix_rotation  # noqa: E402


RESULT_ROOT = WORK_ROOT / "results" / "rarl_v3_geometry_screened_mujoco"
RAW = RESULT_ROOT / "results" / "raw"
PROCESSED = RESULT_ROOT / "results" / "processed"
FIGURES = RESULT_ROOT / "figures"
TABLES = RESULT_ROOT / "paper_tables"

DEVICE = M.DEVICE
DTYPE = M.DTYPE
EPS = M.EPS

# Budgeted settings.  Kept small enough to run locally while preserving the staged logic.
SCREEN_SEEDS = [0, 1, 2]
FINAL_SEEDS = [0, 1, 2]
WARMUP_STEPS = 600
FIELD_BATCH = 96
N_ROT_PROBES = 8
N_REDUCED_PROBES = 8
TRAIN_ITERS = 30
EVAL_EVERY = 5
EVAL_EPISODES = 2
METHODS = ["gda", "egm", "ppm", "nog", "qpg"]
LR = 3e-4
LAMBDA_F = 0.3
SIGMAS = [0.1, 0.3, 0.5]

# Make the reused warmup stack light for a geometry screen.
P.CHUNK = 50
P.CRITIC_UPDATES_PER_CHUNK = 2
P.BATCH_SIZE = 128
P.FIELD_BATCH = FIELD_BATCH
P.CKPT_STEPS = [WARMUP_STEPS]


@dataclass
class Candidate:
    label: str
    env_id: str
    role: str
    b_kind: str
    sigma: float
    env_kwargs: dict[str, Any]

    @property
    def tag(self) -> str:
        s = str(self.sigma).replace(".", "p")
        return f"{self.label}_sigma{s}_{self.b_kind}".replace("-", "_")


def ensure_dirs() -> None:
    for p in [RAW, PROCESSED, FIGURES, TABLES]:
        p.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def try_spec(env_id: str, env_kwargs: dict[str, Any]) -> tuple[M.EnvSpec | None, str]:
    try:
        env = gym.make(env_id, **env_kwargs)
        obs_space = env.observation_space
        act_space = env.action_space
        spec = M.EnvSpec(
            env_id=env_id,
            obs_dim=int(obs_space.shape[0]),
            action_dim=int(act_space.shape[0]),
            action_low=np.asarray(act_space.low, dtype=np.float32),
            action_high=np.asarray(act_space.high, dtype=np.float32),
        )
        env.close()
        return spec, ""
    except Exception as e:
        return None, repr(e)


def b_matrix(kind: str, action_dim: int) -> np.ndarray:
    if kind == "identity":
        return np.eye(action_dim, dtype=np.float32)
    if kind == "swimmer_skew":
        if action_dim != 2:
            raise ValueError("swimmer_skew requires action_dim=2")
        return np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    raise ValueError(kind)


class ActuatorMixChannel:
    name = "actuator_mix"

    def __init__(self, spec: M.EnvSpec, sigma: float, B: np.ndarray) -> None:
        self.spec = spec
        self.sigma = float(sigma)
        self.B = B.astype(np.float32)
        self.clip_count = 0
        self.step_count = 0

    def step(self, env, u_np, w_np):
        if hasattr(env.unwrapped, "data") and hasattr(env.unwrapped.data, "xfrc_applied"):
            env.unwrapped.data.xfrc_applied[:] = 0.0
        mixed = self.B @ np.clip(w_np, -1.0, 1.0)
        raw = u_np + self.sigma * mixed
        env_action = np.clip(raw, self.spec.action_low, self.spec.action_high)
        self.clip_count += int(np.any(np.abs(env_action - raw) > 1e-6))
        self.step_count += 1
        return env.step(env_action.astype(np.float32))

    @property
    def action_clip_fraction(self) -> float:
        return self.clip_count / max(self.step_count, 1)


def candidates() -> list[Candidate]:
    out: list[Candidate] = []
    base = [
        ("Adv-InvertedPendulum-v5", "InvertedPendulum-v5", "main", ["identity"], {}),
        ("Adv-Swimmer-v5", "Swimmer-v5", "main", ["identity", "swimmer_skew"], {}),
        ("Adv-HalfCheetah-LowCost-v5", "HalfCheetah-v5", "main", ["identity"], {"ctrl_cost_weight": 0.0}),
        ("Adv-Hopper-v5", "Hopper-v5", "negative", ["identity"], {}),
        ("Adv-Reacher-v5", "Reacher-v5", "negative", ["identity"], {}),
    ]
    for label, env_id, role, kinds, kwargs in base:
        for sigma in SIGMAS:
            for kind in kinds:
                out.append(Candidate(label, env_id, role, kind, sigma, dict(kwargs)))
    return out


def make_operator(cand: Candidate, seed: int):
    spec, err = try_spec(cand.env_id, cand.env_kwargs)
    if spec is None:
        return None, {"available": 0, "error": err}
    if cand.b_kind == "swimmer_skew" and spec.action_dim != 2:
        return None, {"available": 0, "error": "swimmer_skew requires action_dim=2"}
    actors, z0 = P.make_z0(spec, seed)
    B = b_matrix(cand.b_kind, spec.action_dim)
    channel = ActuatorMixChannel(spec, cand.sigma, B)
    critic, _critic_t, buffer, _ckpts, rng = P.run_warmup(spec, actors, z0, channel, seed, WARMUP_STEPS, env_kwargs=cand.env_kwargs)
    for p in critic.parameters():
        p.requires_grad_(False)
    batch = P.field_batch(buffer, FIELD_BATCH, rng)
    cfg = M.Config(cand.env_id, 0.0, LAMBDA_F, LR, seed, 1)
    game = P.PathwiseCriticGame(spec, cfg, actors, critic)
    game.calibrate_ptau(z0.detach().clone().requires_grad_(True), batch)
    return (spec, actors, z0, critic, buffer, batch, game, rng, channel), {"available": 1}


def reduced_jacobian_metrics(game, z, batch, seed: int) -> dict[str, float]:
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed + 88)
    z_req = z.detach().clone().requires_grad_(True)
    Fv = game.field(z_req, batch, create_graph=True)
    d = z.shape[0]
    probes = []
    for _ in range(N_REDUCED_PROBES):
        v = torch.randn(d, generator=gen, dtype=DTYPE, device=DEVICE)
        v = v / (torch.linalg.norm(v) + EPS)
        probes.append(v)
    Jred = np.zeros((len(probes), len(probes)), dtype=np.float64)
    for j, v in enumerate(probes):
        Jv = M.jvp_field(game, z_req, batch, v).detach()
        for i, u in enumerate(probes):
            Jred[i, j] = float(torch.dot(u, Jv).item())
    eig = np.linalg.eigvals(Jred)
    imag = np.abs(np.imag(eig))
    complex_frac = float(np.mean(imag > 1e-6))
    return {
        "reduced_complex_frac": complex_frac,
        "reduced_max_imag": float(np.max(imag)) if len(imag) else math.nan,
        "reduced_gradient_proxy": float(np.linalg.norm(np.diag(Jred)) / (np.linalg.norm(Jred) + EPS)),
    }


def qp_probe_metrics(game, z, batch) -> dict[str, float]:
    z_req = z.detach().clone().requires_grad_(True)
    Fv = game.field(z_req, batch, create_graph=True).detach()
    Gv = M.jvp_field(game, z_req, batch, Fv).detach()
    coeff = M.quadratic_coefficients(game, z_req, batch, Fv, Gv, beta_max=LR, gamma_max=LR, signed_box=False)
    beta, gamma = float(coeff["best"][0]), float(coeff["best"][1])
    nog_beta = float(coeff["nog"][0])
    g_contrib = abs(gamma) * float(torch.linalg.norm(Gv).item()) / (
        abs(beta) * float(torch.linalg.norm(Fv).item()) + abs(gamma) * float(torch.linalg.norm(Gv).item()) + EPS
    )
    v0 = float(game.merit_tensor(z_req, batch, create_graph=True).detach().item())
    z_nog = z.detach() + nog_beta * (-Fv)
    z_qp = z.detach() + beta * (-Fv) + gamma * Gv
    vn = float(game.merit_tensor(z_nog.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
    vq = float(game.merit_tensor(z_qp.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
    return {
        "active_gamma_frac": float(gamma > 1e-12),
        "fallback_probe_frac": float((not math.isfinite(vq)) or vq > vn + 1e-8),
        "G_contribution_ratio": g_contrib,
        "beta": beta,
        "gamma": gamma,
        "gamma_beta_ratio": gamma / (abs(beta) + EPS),
        "same_batch_V0": v0,
        "same_batch_V_noG": vn,
        "same_batch_V_QPG": vq,
    }


def geometry_for(cand: Candidate, seed: int) -> dict[str, Any]:
    start = time.perf_counter()
    op, meta = make_operator(cand, seed)
    row: dict[str, Any] = {
        "candidate": cand.label,
        "env_id": cand.env_id,
        "role": cand.role,
        "sigma": cand.sigma,
        "B": cand.b_kind,
        "seed": seed,
        **meta,
    }
    if op is None:
        return row
    spec, actors, z0, critic, _buffer, batch, game, _rng, channel = op
    try:
        gm = M.geometry_metrics(game, z0.detach().clone().requires_grad_(True), batch)
        rm = matrix_rotation(game, z0, batch, n_probes=N_ROT_PROBES, seed=seed)
        with torch.no_grad():
            obs = batch["obs"]
            mp = P.mu(actors, z0, obs, "p")
            ma = P.mu(actors, z0, obs, "a")
        qmix = P.qc_mixed_norms(critic, obs, mp, ma)
        red = reduced_jacobian_metrics(game, z0, batch, seed)
        qp = qp_probe_metrics(game, z0, batch)
        row.update(
            {
                "obs_dim": spec.obs_dim,
                "action_dim": spec.action_dim,
                "rotation_ratio_proxy": rm,
                "vector_rotation_ratio_proxy": gm["rotation_ratio_joint"],
                "cos_F_G": gm["cos_F_G"],
                "non_collinearity": 1.0 - abs(gm["cos_F_G"]),
                "varrho2_over_Phi_proxy": gm["varrho2_over_Phi_proxy"],
                "field_norm": float(torch.linalg.norm(game.field(z0.detach().clone().requires_grad_(True), batch, create_graph=False)).item()),
                "action_clip_fraction": channel.action_clip_fraction,
                **qmix,
                **red,
                **qp,
                "wall_clock_sec": time.perf_counter() - start,
            }
        )
    except Exception as e:
        row.update({"available": 0, "error": repr(e)})
    return row


def eval_policy(spec: M.EnvSpec, cand: Candidate, actors, z, seed: int, br_z=None, episodes: int = EVAL_EPISODES) -> dict[str, float]:
    B = b_matrix(cand.b_kind, spec.action_dim)
    vals_clean, vals_robust = [], []
    for ep in range(episodes):
        for mode in ["clean", "robust"]:
            env = gym.make(cand.env_id, **cand.env_kwargs)
            obs, _ = env.reset(seed=seed + 1000 * ep + (0 if mode == "clean" else 500))
            total = 0.0
            done = False
            steps = 0
            while not done and steps < 1000:
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), dtype=DTYPE, device=DEVICE)
                with torch.no_grad():
                    u = P.mu(actors, z, obs_t, "p").squeeze(0).cpu().numpy()
                    if mode == "clean":
                        v = np.zeros_like(u)
                    else:
                        zz = br_z if br_z is not None else z
                        v = P.mu(actors, zz, obs_t, "a").squeeze(0).cpu().numpy()
                a = np.clip(u + cand.sigma * (B @ v), spec.action_low, spec.action_high)
                obs, rew, terminated, truncated, _ = env.step(a.astype(np.float32))
                total += float(rew)
                done = bool(terminated or truncated)
                steps += 1
            env.close()
            (vals_clean if mode == "clean" else vals_robust).append(total)
    clean = float(np.mean(vals_clean))
    robust = float(np.mean(vals_robust))
    return {
        "nominal_return": clean,
        "robust_return": robust,
        "robust_degradation": clean - robust,
        "adversary_return": -robust,
    }


def br_refine(actors, game, z, batch, steps: int = 5, lr: float = 1e-3):
    br = z.detach().clone()
    psi0 = br[actors.a_slice].detach().clone()
    for _ in range(steps):
        br_req = br.detach().clone().requires_grad_(True)
        J = game.J_surr(br_req, batch)
        grad = torch.autograd.grad(J, br_req)[0]
        br = br.detach()
        br[actors.a_slice] = br[actors.a_slice] - lr * grad[actors.a_slice]
        diff = br[actors.a_slice] - psi0
        n = torch.linalg.norm(diff)
        if n > 0.5:
            br[actors.a_slice] = psi0 + diff / n * 0.5
    return br.detach()


def method_step(game, z, batch, method: str):
    z_req = z.detach().clone().requires_grad_(True)
    Fv = game.field(z_req, batch, create_graph=True)
    Fd = Fv.detach()
    if method == "gda":
        return (z - LR * Fd).detach(), {"oracle_calls": 1, "gamma": 0.0, "beta": LR, "fallback": 0, "G_contrib": 0.0}
    if method == "egm":
        zh = (z - LR * Fd).detach()
        Fh = game.field(zh.detach().clone().requires_grad_(True), batch, create_graph=False).detach()
        return (z - LR * Fh).detach(), {"oracle_calls": 2, "gamma": 0.0, "beta": LR, "fallback": 0, "G_contrib": 0.0}
    if method == "ppm":
        zi = z.detach()
        for _ in range(3):
            Fi = game.field(zi.detach().clone().requires_grad_(True), batch, create_graph=False).detach()
            zi = (z - LR * Fi).detach()
        return zi, {"oracle_calls": 3, "gamma": 0.0, "beta": LR, "fallback": 0, "G_contrib": 0.0}
    Gv = M.jvp_field(game, z_req, batch, Fd).detach()
    coeff = M.quadratic_coefficients(game, z_req, batch, Fd, Gv, beta_max=LR, gamma_max=LR, signed_box=False)
    beta_n = float(coeff["nog"][0])
    z_n = (z + beta_n * (-Fd)).detach()
    if method == "nog":
        return z_n, {"oracle_calls": 2, "gamma": 0.0, "beta": beta_n, "fallback": 0, "G_contrib": 0.0}
    beta, gamma = float(coeff["best"][0]), float(coeff["best"][1])
    z_q = (z + beta * (-Fd) + gamma * Gv).detach()
    vn = float(game.merit_tensor(z_n.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
    vq = float(game.merit_tensor(z_q.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
    fallback = int((not math.isfinite(vq)) or vq > vn + 1e-8)
    if fallback:
        z_q = z_n
        gamma = 0.0
    g_contrib = abs(gamma) * float(torch.linalg.norm(Gv).item()) / (
        abs(beta) * float(torch.linalg.norm(Fd).item()) + abs(gamma) * float(torch.linalg.norm(Gv).item()) + EPS
    )
    return z_q, {"oracle_calls": 3, "gamma": gamma, "beta": beta, "fallback": fallback, "G_contrib": g_contrib}


def train_setting(cand: Candidate, seed: int, method: str) -> list[dict[str, Any]]:
    op, meta = make_operator(cand, seed)
    if op is None:
        return []
    spec, actors, z0, _critic, buffer, batch, game, rng, _channel = op
    z = z0.detach().clone()
    rows = []
    start = time.perf_counter()
    for it in range(TRAIN_ITERS + 1):
        cur_batch = P.field_batch(buffer, FIELD_BATCH, rng)
        metrics = game.metrics(z, cur_batch, geometry=False)
        evals = eval_policy(spec, cand, actors, z, seed=seed * 10000 + it)
        br_z = br_refine(actors, game, z, cur_batch)
        br_eval = eval_policy(spec, cand, actors, z, seed=seed * 20000 + it, br_z=br_z, episodes=1)
        rows.append(
            {
                "candidate": cand.label,
                "env_id": cand.env_id,
                "sigma": cand.sigma,
                "B": cand.b_kind,
                "seed": seed,
                "method": method,
                "iteration": it,
                "wall_clock_sec": time.perf_counter() - start,
                "V": metrics["V"],
                "field_norm": metrics["field_norm"],
                "P_tau": metrics["P_tau"],
                "robust_best_response_return": br_eval["robust_return"],
                **evals,
            }
        )
        if it == TRAIN_ITERS:
            break
        z, info = method_step(game, z, cur_batch, method)
        rows[-1].update(
            {
                "gamma": info["gamma"],
                "beta": info["beta"],
                "gamma_beta_ratio": info["gamma"] / (abs(info["beta"]) + EPS),
                "active_gamma": float(abs(info["gamma"]) > 1e-12),
                "fallback": info["fallback"],
                "G_contribution_ratio": info["G_contrib"],
                "oracle_calls": info["oracle_calls"],
            }
        )
    path = RAW / f"mujoco_{cand.tag}_{method}_seed{seed}.csv"
    M.write_csv(path, rows)
    return rows


def auc(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.trapz(vals, dx=1.0)) if vals else math.nan


def summarize_training(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["candidate"], r["env_id"], r["sigma"], r["B"], r["method"], r["seed"]), []).append(r)
    out = []
    for key, vals in groups.items():
        vals = sorted(vals, key=lambda x: int(x["iteration"]))
        out.append(
            {
                "candidate": key[0],
                "env_id": key[1],
                "sigma": key[2],
                "B": key[3],
                "method": key[4],
                "seed": key[5],
                "V_AUC": auc([v["V"] for v in vals]),
                "field_norm_AUC": auc([v["field_norm"] for v in vals]),
                "P_tau_AUC": auc([v["P_tau"] for v in vals]),
                "robust_br_return_AUC": auc([v["robust_best_response_return"] for v in vals]),
                "robust_degradation_AUC": auc([v["robust_degradation"] for v in vals]),
                "nominal_return_AUC": auc([v["nominal_return"] for v in vals]),
                "active_gamma_frac": float(np.mean([float(v.get("active_gamma", 0.0) or 0.0) for v in vals])),
                "fallback_frac": float(np.mean([float(v.get("fallback", 0.0) or 0.0) for v in vals])),
                "G_contribution_ratio": float(np.mean([float(v.get("G_contribution_ratio", 0.0) or 0.0) for v in vals])),
                "wall_clock_sec": vals[-1]["wall_clock_sec"],
                "oracle_calls_mean": float(np.mean([float(v.get("oracle_calls", 0.0) or 0.0) for v in vals])),
            }
        )
    return out


def make_figures(geom: list[dict[str, Any]], curves: list[dict[str, Any]], train_summary: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    gdf = pd.DataFrame(geom)
    ok = gdf[gdf.get("available", 0).astype(str) == "1"] if not gdf.empty else gdf
    if not ok.empty:
        labels = ok["candidate"].astype(str) + "\nσ=" + ok["sigma"].astype(str) + "\n" + ok["B"].astype(str)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes[0, 0].bar(range(len(ok)), ok["rotation_ratio_proxy"].astype(float))
        axes[0, 0].set_title("Rotation ratio proxy")
        axes[0, 1].bar(range(len(ok)), ok["cos_F_G"].astype(float))
        axes[0, 1].set_title("cos(F,G)")
        axes[1, 0].bar(range(len(ok)), ok["non_collinearity"].astype(float))
        axes[1, 0].set_title("Non-collinearity 1-|cos|")
        axes[1, 1].bar(range(len(ok)), ok["reduced_complex_frac"].astype(float))
        axes[1, 1].set_title("Reduced complex eigenmode fraction")
        for ax in axes.flat:
            ax.set_xticks(range(len(ok)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "mujoco_geometry_screening.png", dpi=180)
        fig.savefig(FIGURES / "mujoco_geometry_screening.pdf")
        plt.close(fig)

    cdf = pd.DataFrame(curves)
    sdf = pd.DataFrame(train_summary)
    if not cdf.empty:
        chosen = cdf[["candidate", "sigma", "B"]].drop_duplicates().head(1).iloc[0]
        sub = cdf[(cdf["candidate"] == chosen["candidate"]) & (cdf["sigma"] == chosen["sigma"]) & (cdf["B"] == chosen["B"])]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for method in METHODS:
            m = sub[sub["method"] == method]
            if m.empty:
                continue
            grouped = m.groupby("iteration", as_index=False).mean(numeric_only=True)
            axes[0, 0].plot(grouped["iteration"], grouped["V"], label=method)
            axes[0, 1].plot(grouped["iteration"], grouped["field_norm"], label=method)
            axes[1, 0].plot(grouped["iteration"], grouped["robust_best_response_return"], label=method)
            axes[1, 1].plot(grouped["iteration"], grouped["robust_degradation"], label=method)
        axes[0, 0].set_title("Composite Lyapunov V")
        axes[0, 1].set_title("Field norm ||F||")
        axes[1, 0].set_title("Robust BR return")
        axes[1, 1].set_title("Robust degradation")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "mujoco_rarl_main.png", dpi=180)
        fig.savefig(FIGURES / "mujoco_rarl_main.pdf")
        plt.close(fig)

    if not ok.empty and not sdf.empty:
        gain_rows = []
        for (cand, sig, b), grp in sdf.groupby(["candidate", "sigma", "B"]):
            mean = grp.groupby("method", as_index=False).mean(numeric_only=True)
            if "qpg" in set(mean["method"]) and "nog" in set(mean["method"]):
                q = mean[mean["method"] == "qpg"].iloc[0]
                n = mean[mean["method"] == "nog"].iloc[0]
                ge = ok[(ok["candidate"] == cand) & (ok["sigma"].astype(float) == float(sig)) & (ok["B"] == b)]
                if not ge.empty:
                    gain_rows.append(
                        {
                            "rotation": float(ge["rotation_ratio_proxy"].median()),
                            "non_collinearity": float(ge["non_collinearity"].median()),
                            "gain_qpg_over_nog": (float(q["robust_br_return_AUC"]) - float(n["robust_br_return_AUC"])) / (abs(float(n["robust_br_return_AUC"])) + EPS),
                        }
                    )
        if gain_rows:
            gd = pd.DataFrame(gain_rows)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(gd["rotation"], gd["gain_qpg_over_nog"], s=80)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("rotation ratio")
            ax.set_ylabel("QP+G gain over noG (robust BR AUC)")
            ax.set_title("Gain versus skew geometry")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES / "mujoco_gain_vs_skew.png", dpi=180)
            plt.close(fig)


def write_tables(geom: list[dict[str, Any]], training: list[dict[str, Any]], decision: str) -> None:
    import pandas as pd

    TABLES.mkdir(parents=True, exist_ok=True)
    gdf = pd.DataFrame(geom)
    if not gdf.empty:
        cols = ["candidate", "sigma", "B", "rotation_ratio_proxy", "cos_F_G", "non_collinearity", "reduced_complex_frac", "G_contribution_ratio", "screen_label"]
        gdf[[c for c in cols if c in gdf.columns]].to_latex(TABLES / "table_mujoco_geometry.tex", index=False, float_format="%.3g")
    tdf = pd.DataFrame(training)
    if not tdf.empty:
        cols = ["candidate", "sigma", "B", "method", "robust_br_return_AUC", "robust_degradation_AUC", "V_AUC", "active_gamma_frac", "fallback_frac"]
        tdf[[c for c in cols if c in tdf.columns]].to_latex(TABLES / "table_mujoco_rarl.tex", index=False, float_format="%.3g")
    write_text(
        TABLES / "mujoco_subsection_text.md",
        "\n".join(
            [
                "# Geometry-screened MuJoCo RARL wrappers",
                "",
                "We do not claim universal improvement on standard MuJoCo benchmarks. Instead, we test whether the field-curvature mechanism survives neural policies in MuJoCo-based zero-sum RARL wrappers. The wrappers use adversarial action perturbations rather than direct bilinear reward injection. We first measure the induced saddle-field geometry and then evaluate QP+G only in regimes where the diagnostics indicate skew dominance.",
                "",
                "Wrappers whose diagnostics are nearly potential-like are reported as negative controls rather than failures: the theory predicts that the curvature direction should not provide a strong field-energy advantage when the symmetric action dominates.",
                "",
                f"Decision: `{decision}`",
            ]
        ),
    )


def label_geometry(row: dict[str, Any]) -> str:
    if int(row.get("available", 0)) != 1:
        return "UNAVAILABLE"
    rot = float(row.get("rotation_ratio_proxy", 0.0))
    noncol = float(row.get("non_collinearity", 0.0))
    cfrac = float(row.get("reduced_complex_frac", 0.0))
    gcon = float(row.get("G_contribution_ratio", 0.0))
    fb = float(row.get("fallback_probe_frac", 1.0))
    if rot >= 0.5 and noncol >= 0.4 and cfrac > 0.0 and gcon >= 0.05 and fb <= 0.5:
        return "SKEW_PROMISING"
    if rot >= 0.25 and noncol >= 0.25:
        return "MODERATE_SKEW_DIAGNOSTIC"
    return "LOW_SKEW_NEGATIVE_DIAGNOSTIC"


def main() -> None:
    ensure_dirs()
    print("=== RARL-v3 geometry-screened neural MuJoCo RARL ===", flush=True)
    geom_rows: list[dict[str, Any]] = []
    for cand in candidates():
        for seed in SCREEN_SEEDS:
            print(f"[screen] {cand.tag} seed={seed}", flush=True)
            row = geometry_for(cand, seed)
            row["screen_label"] = label_geometry(row)
            geom_rows.append(row)
            M.write_csv(RAW / "mujoco_geometry_diagnostics.csv", geom_rows)

    import pandas as pd

    gdf = pd.DataFrame(geom_rows)
    M.write_csv(RAW / "mujoco_geometry_diagnostics.csv", geom_rows)
    summary_rows: list[dict[str, Any]] = []
    if not gdf.empty:
        for keys, grp in gdf.groupby(["candidate", "env_id", "role", "sigma", "B"], dropna=False):
            numeric_cols = [c for c in grp.columns if c not in {"candidate", "env_id", "role", "B", "error", "screen_label"}]
            med = grp[numeric_cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True).to_dict()
            labels = grp["screen_label"].tolist() if "screen_label" in grp else []
            label = "SKEW_PROMISING" if labels.count("SKEW_PROMISING") >= max(1, len(labels) // 2 + 1) else (
                "MODERATE_SKEW_DIAGNOSTIC" if any(l == "MODERATE_SKEW_DIAGNOSTIC" for l in labels) else (
                    "UNAVAILABLE" if all(l == "UNAVAILABLE" for l in labels) else "LOW_SKEW_NEGATIVE_DIAGNOSTIC"
                )
            )
            summary_rows.append({"candidate": keys[0], "env_id": keys[1], "role": keys[2], "sigma": keys[3], "B": keys[4], "screen_label": label, **med})
    M.write_csv(PROCESSED / "mujoco_geometry_summary.csv", summary_rows)

    promising = [r for r in summary_rows if r["screen_label"] == "SKEW_PROMISING"]
    # If no strong cell exists, run one best moderate cell only as a reference negative-control curve.
    if not promising:
        ranked = sorted(
            [r for r in summary_rows if r["screen_label"] != "UNAVAILABLE"],
            key=lambda r: (float(r.get("rotation_ratio_proxy", 0.0)), float(r.get("non_collinearity", 0.0))),
            reverse=True,
        )
        promising = ranked[:1]

    curve_rows: list[dict[str, Any]] = []
    if promising:
        row = promising[0]
        selected = Candidate(str(row["candidate"]), str(row["env_id"]), str(row.get("role", "selected")), str(row["B"]), float(row["sigma"]), {} if "HalfCheetah" not in str(row["candidate"]) else {"ctrl_cost_weight": 0.0})
        print(f"[train] selected {selected.tag} label={row['screen_label']}", flush=True)
        for seed in FINAL_SEEDS:
            for method in METHODS:
                print(f"  [train] {method} seed={seed}", flush=True)
                curve_rows.extend(train_setting(selected, seed, method))
    train_summary = summarize_training(curve_rows)
    M.write_csv(PROCESSED / "mujoco_training_summary.csv", train_summary)
    make_figures(geom_rows, curve_rows, train_summary)

    decision = "NO_SKEW_DOMINATED_SETTING_FOUND"
    if any(r["screen_label"] == "SKEW_PROMISING" for r in summary_rows) and train_summary:
        tdf = pd.DataFrame(train_summary)
        mean = tdf.groupby("method", as_index=False).mean(numeric_only=True)
        if {"qpg", "nog"}.issubset(set(mean["method"])):
            q = mean[mean["method"] == "qpg"].iloc[0]
            n = mean[mean["method"] == "nog"].iloc[0]
            gain = (float(q["robust_br_return_AUC"]) - float(n["robust_br_return_AUC"])) / (abs(float(n["robust_br_return_AUC"])) + EPS)
            decision = "QPG_GAIN_IN_SKEW_SCREENED_SETTING" if gain > 0.05 else "SKEW_SCREENED_BUT_QPG_NO_CLEAR_GAIN"
    write_tables(geom_rows, train_summary, decision)

    low = [r for r in summary_rows if r["screen_label"] == "LOW_SKEW_NEGATIVE_DIAGNOSTIC"]
    skew = [r for r in summary_rows if r["screen_label"] == "SKEW_PROMISING"]
    lines = [
        "# RARL-v3 Geometry-Screened MuJoCo RARL Report",
        "",
        f"- decision: `{decision}`",
        f"- geometry rows: `{len(geom_rows)}`",
        f"- skew-promising settings: `{len(skew)}`",
        f"- low-skew negative diagnostics: `{len(low)}`",
        "",
        "## Skew-Promising Settings",
    ]
    if skew:
        for r in skew:
            lines.append(f"- `{r['candidate']}` sigma=`{r['sigma']}` B=`{r['B']}` rotation=`{float(r.get('rotation_ratio_proxy', 0.0)):.3f}` noncol=`{float(r.get('non_collinearity', 0.0)):.3f}`")
    else:
        lines.append("- None met the strict `SKEW_PROMISING` screen. The best available cell was trained only as a reference diagnostic, not as a main positive claim.")
    lines.extend(["", "## Negative Diagnostics"])
    for r in sorted(low, key=lambda x: float(x.get("rotation_ratio_proxy", 0.0)), reverse=True)[:10]:
        lines.append(f"- `{r['candidate']}` sigma=`{r['sigma']}` B=`{r['B']}` rotation=`{float(r.get('rotation_ratio_proxy', 0.0)):.3f}`: nearly potential-like; this is consistent with Theorem 1(b).")
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- raw diagnostics: `{RAW / 'mujoco_geometry_diagnostics.csv'}`",
            f"- geometry summary: `{PROCESSED / 'mujoco_geometry_summary.csv'}`",
            f"- training summary: `{PROCESSED / 'mujoco_training_summary.csv'}`",
            f"- geometry figure: `{FIGURES / 'mujoco_geometry_screening.png'}`",
            f"- main figure: `{FIGURES / 'mujoco_rarl_main.png'}`",
            "",
            "## Wording Guardrail",
            "This is a geometry-screened MuJoCo RARL wrapper study, not a universal MuJoCo benchmark-improvement claim. No direct bilinear reward term was used in the main experiment.",
        ]
    )
    write_text(RESULT_ROOT / "final_report.md", "\n".join(lines))
    print(f"[done] {decision}", flush=True)


if __name__ == "__main__":
    main()
