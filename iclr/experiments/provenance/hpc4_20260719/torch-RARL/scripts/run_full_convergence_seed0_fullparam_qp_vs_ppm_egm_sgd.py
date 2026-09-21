from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.RARL import RARL
from scripts.full_policy_followup_common import SavedRun, load_yaml, make_eval_vec_env, set_rarl_eval_mode
from scripts.run_perflyap_stage412_actual_merit_oracle import rollout_return_with_model


@dataclass(frozen=True)
class MethodSpec:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]
    family: str


@dataclass
class RunRecord:
    method: str
    budget: int
    status: str
    latest_run_dir: Optional[pathlib.Path]
    analysis_dir: pathlib.Path
    run_root: pathlib.Path
    error_message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Full convergence seed0: proposed QP vs PPM/EGM/SGD")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--initial-iterations", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=51200)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--short-horizon", type=int, default=16)
    parser.add_argument("--short-return-episodes", type=int, default=1)
    parser.add_argument("--robustness-strengths", type=str, default="0,0.25,0.5,1,2,4")
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def render_kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens: List[str] = []
    for key, value in sorted(kwargs.items()):
        rendered = repr(value) if isinstance(value, str) else value
        tokens.append(f"{key}:{rendered}")
    return tokens


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def build_saved_run(run_dir: pathlib.Path, method: str) -> SavedRun:
    model_dir = next(path for path in run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists())
    return SavedRun(
        method=method,
        tag=method,
        run_root=run_dir,
        model_dir=model_dir,
        args_data=load_yaml(model_dir / "args.yml"),
        config_data=load_yaml(model_dir / "config.yml"),
    )


def first_existing(frame: pd.DataFrame, names: Sequence[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"Missing columns {list(names)} in {list(frame.columns)}")


def compute_auc(x: pd.Series, y: pd.Series) -> float:
    xs = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    xs = xs[mask]
    ys = ys[mask]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.trapz(ys, xs))


def moving_average(values: pd.Series, window: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), int(arr.size)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def aggregate_series(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    if df.empty or value_col not in df.columns:
        return {
            "final": float("nan"),
            "last5_mean": float("nan"),
            "prev5_mean": float("nan"),
            "auc": float("nan"),
            "slope_last5": float("nan"),
        }
    sdf = df.sort_values("outer_iteration")
    vals = pd.to_numeric(sdf[value_col], errors="coerce")
    xs = pd.to_numeric(sdf["outer_iteration"], errors="coerce")
    prev5 = float(vals.iloc[-10:-5].mean()) if len(vals) >= 10 else float("nan")
    if len(vals) >= 3:
        tail_x = xs.tail(min(5, len(xs))).to_numpy(dtype=np.float64)
        tail_y = vals.tail(min(5, len(vals))).to_numpy(dtype=np.float64)
        mask = np.isfinite(tail_x) & np.isfinite(tail_y)
        if mask.sum() >= 2:
            slope_last5 = float(np.polyfit(tail_x[mask], tail_y[mask], deg=1)[0])
        else:
            slope_last5 = float("nan")
    else:
        slope_last5 = float("nan")
    return {
        "final": float(vals.iloc[-1]) if len(vals) else float("nan"),
        "last5_mean": float(vals.tail(5).mean()) if len(vals) else float("nan"),
        "prev5_mean": prev5,
        "auc": compute_auc(sdf["outer_iteration"], vals),
        "slope_last5": slope_last5,
    }


def build_candidates() -> List[MethodSpec]:
    proposed_common = {
        "perflyap_scope": "full_policy_actor_weighted",
        "lambda_N": 0.0,
        "lambda_P": 1.0,
        "lambda_critic": 1.0,
        "logstd_weight": 0.0,
        "use_scale_normalization": False,
        "qp_fd_eps": 1e-3,
        "qp_beta_probe": 1e-3,
        "qp_gamma_probe": 1e-6,
        "qp_ridge": 1e-8,
        "qp_rho": 1e-8,
        "qp_beta_max": 3e-2,
        "qp_gamma_max": 3e-5,
        "qp_max_update_norm": 0.005,
        "qp_eps": 1e-8,
        "eta_egm_reference": 1e-3,
        "direction_mode": "egm_minus_JF_F",
        "selector_mode": "safe_fixed_minusg",
        "selector_beta_grid": "0,0.009,0.012,0.015",
        "selector_gamma_grid": "0,1.2e-05,2.4e-05,3e-05",
        "ls_fit_variant": "LS_all_grid",
        "short_return_horizon": 16,
        "short_return_episodes": 1,
        "short_return_seed_offset": 0,
        "cost_mode": "mixed_clean_unclipped_actor_surrogate_cost",
        "fixed_beta_raw": 0.012,
        "fixed_gamma_raw": 2.4e-05,
    }
    return [
        MethodSpec(
            method="proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG",
            optimizer="proposed_qp_perfLyap",
            lr=1.0,
            max_grad_norm=1.0,
            vf_coef=1.0,
            optimizer_kwargs=proposed_common,
            family="proposed",
        ),
        MethodSpec(
            method="ppm_lr5e-3",
            optimizer="ppm",
            lr=5e-3,
            max_grad_norm=1.0,
            vf_coef=1.0,
            optimizer_kwargs={"inner_steps": 10},
            family="baseline",
        ),
        MethodSpec(
            method="egm_lr5e-3",
            optimizer="egm",
            lr=5e-3,
            max_grad_norm=1.0,
            vf_coef=1.0,
            optimizer_kwargs={},
            family="baseline",
        ),
        MethodSpec(
            method="sgd_lr1e-3",
            optimizer="sgd",
            lr=1e-3,
            max_grad_norm=1.0,
            vf_coef=1.0,
            optimizer_kwargs={},
            family="baseline",
        ),
    ]


def analyze_run(repo_dir: pathlib.Path, python_path: str, latest_run_dir: pathlib.Path, analysis_dir: pathlib.Path, method: str) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_root = analysis_dir.parent
    for name in ("stdout.txt", "stderr.txt"):
        source = run_root / name
        target = analysis_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    run_command(
        [python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", method],
        cwd=repo_dir,
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )


def ensure_run_for_budget(candidate: MethodSpec, args: argparse.Namespace, budget_root: pathlib.Path, iterations: int) -> RunRecord:
    run_root = budget_root / "runs_seed0" / candidate.method
    saved_models_dir = run_root / "saved_models"
    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    analysis_dir = run_root / "analysis"
    run_root.mkdir(parents=True, exist_ok=True)

    env_root = saved_models_dir / "rarl-ppo" / args.env
    try:
        if env_root.exists():
            latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
            if not (analysis_dir / "run_summary.csv").exists():
                analyze_run(pathlib.Path(args.repo_dir), args.python_path, latest_run_dir, analysis_dir, candidate.method)
            return RunRecord(candidate.method, iterations, "completed", latest_run_dir, analysis_dir, run_root)

        command = [
            args.python_path,
            "scripts/train_adversary.py",
            "--algo",
            "rarl",
            "--rarl-config",
            "ppo",
            "--env",
            args.env,
            "--adv-impact",
            "control",
            "--device",
            args.device,
            "--verbose",
            "1",
            "--seed",
            str(args.seed),
            "-n",
            str(iterations),
            "--eval-freq",
            str(args.eval_freq),
            "--save-freq",
            str(args.eval_freq),
            "--n-eval-episodes",
            str(args.n_eval_episodes),
            "--saved-models-path",
            str(saved_models_dir),
            "--log-folder",
            str(logging_dir),
            "--tensorboard-log",
            str(tb_dir),
            "--protagonist-policy",
            "MlpPolicy",
            "--adversary-policy",
            "MlpPolicy",
            "--protagonist-optimizer",
            candidate.optimizer,
            "--adversary-optimizer",
            candidate.optimizer,
            "--protagonist-lr",
            str(candidate.lr),
            "--adversary-lr",
            str(candidate.lr),
            "--protagonist-max-grad-norm",
            str(candidate.max_grad_norm),
            "--adversary-max-grad-norm",
            str(candidate.max_grad_norm),
            "--protagonist-vf-coef",
            str(candidate.vf_coef),
            "--adversary-vf-coef",
            str(candidate.vf_coef),
        ]
        kwargs_tokens = render_kwargs_tokens(candidate.optimizer_kwargs)
        if kwargs_tokens:
            command.extend(["--protagonist-optimizer-kwargs", *kwargs_tokens])
            command.extend(["--adversary-optimizer-kwargs", *kwargs_tokens])
        run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
        latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
        analyze_run(pathlib.Path(args.repo_dir), args.python_path, latest_run_dir, analysis_dir, candidate.method)
        return RunRecord(candidate.method, iterations, "completed", latest_run_dir, analysis_dir, run_root)
    except Exception as exc:  # noqa: BLE001
        latest_run_dir = None
        if env_root.exists():
            try:
                latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
                if latest_run_dir is not None and not (analysis_dir / "run_summary.csv").exists():
                    analyze_run(pathlib.Path(args.repo_dir), args.python_path, latest_run_dir, analysis_dir, candidate.method)
            except Exception:
                latest_run_dir = None
        return RunRecord(candidate.method, iterations, "failed", latest_run_dir, analysis_dir, run_root, error_message=str(exc))


def read_training_frame(run_root: pathlib.Path, latest_run_dir: pathlib.Path, method: str) -> pd.DataFrame:
    analysis_dir = run_root / "analysis"
    args_path = next(path for path in latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists()) / "args.yml"
    args_data = load_yaml(args_path)
    n_steps = int(args_data.get("n_steps", 2048))
    df = pd.read_csv(analysis_dir / "training_episode_returns.csv").copy()
    time_col = first_existing(df, ["timesteps", "timestep", "total_timesteps", "cumulative_timesteps"])
    ret_col = first_existing(df, ["episode_return", "reward", "ep_rew_mean", "return"])
    df["timestep"] = pd.to_numeric(df[time_col], errors="coerce")
    df["episode_return"] = pd.to_numeric(df[ret_col], errors="coerce")
    df["outer_iteration"] = df["timestep"] / float(n_steps)
    df["method"] = method
    return df


def read_eval_frame(path: pathlib.Path, method: str, n_steps: int, eval_type: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    time_col = first_existing(df, ["timesteps", "timestep", "total_timesteps"])
    mean_col = first_existing(df, ["mean_reward", "clean_mean", "adv_mean", "adversarial_mean", "control_adv_mean", "eval_mean_reward"])
    std_col = None
    for name in ["std_reward", "clean_std", "adv_std", "adversarial_std", "control_adv_std", "eval_std_reward"]:
        if name in df.columns:
            std_col = name
            break
    df["timestep"] = pd.to_numeric(df[time_col], errors="coerce")
    df["mean_reward"] = pd.to_numeric(df[mean_col], errors="coerce")
    df["std_reward"] = pd.to_numeric(df[std_col], errors="coerce") if std_col else np.nan
    df["outer_iteration"] = df["timestep"] / float(n_steps)
    df["method"] = method
    df["eval_type"] = eval_type
    if "applied_perturbation_norm" not in df.columns:
        df["applied_perturbation_norm"] = np.nan
    if "clip_fraction_eval" not in df.columns:
        df["clip_fraction_eval"] = np.nan
    return df


def read_param_frame(run_root: pathlib.Path, latest_run_dir: pathlib.Path, method: str) -> pd.DataFrame:
    analysis_dir = run_root / "analysis"
    args_path = next(path for path in latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists()) / "args.yml"
    args_data = load_yaml(args_path)
    n_steps = int(args_data.get("n_steps", 2048))
    df = pd.read_csv(analysis_dir / "parameter_norms.csv").copy()
    time_col = first_existing(df, ["num_timesteps", "timesteps", "timestep"])
    df["num_timesteps"] = pd.to_numeric(df[time_col], errors="coerce")
    df["outer_iteration"] = df["num_timesteps"] / float(n_steps)
    df["method"] = method
    return df


def recover_update_frames(records: Sequence[RunRecord]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for record in records:
        if record.latest_run_dir is None or not record.analysis_dir.exists():
            continue
        args_path = next(path for path in record.latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists()) / "args.yml"
        args_data = load_yaml(args_path)
        n_steps = int(args_data.get("n_steps", 2048))
        metrics_root = record.latest_run_dir / "analysis"
        summary_path = record.analysis_dir / "run_summary.csv"
        summary_row = pd.read_csv(summary_path).iloc[0].to_dict() if summary_path.exists() else {}
        for role in ("protagonist", "adversary"):
            path = metrics_root / f"{role}_training_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).copy()
            if df.empty:
                continue
            df["method"] = record.method
            df["optimizer_role"] = role
            df["num_timesteps"] = pd.to_numeric(df[first_existing(df, ["num_timesteps", "timesteps", "timestep"])], errors="coerce")
            df["outer_iteration"] = df["num_timesteps"] / float(n_steps)
            df["lr"] = float(summary_row.get("protagonist_lr" if role == "protagonist" else "adversary_lr", np.nan))
            df["max_grad_norm"] = float(summary_row.get("protagonist_max_grad_norm" if role == "protagonist" else "adversary_max_grad_norm", np.nan))
            df["vf_coef"] = float(summary_row.get("protagonist_vf_coef" if role == "protagonist" else "adversary_vf_coef", np.nan))
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def recover_qp_diagnostics(records: Sequence[RunRecord]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for record in records:
        if record.method != "proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG":
            continue
        if record.latest_run_dir is None:
            continue
        args_path = next(path for path in record.latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists()) / "args.yml"
        args_data = load_yaml(args_path)
        n_steps = int(args_data.get("n_steps", 2048))
        metrics_root = record.latest_run_dir / "analysis"
        for role, diag_name in (
            ("protagonist", "protagonist_proposed_qp_perflyap_diagnostics.csv"),
            ("adversary", "adversary_proposed_qp_perflyap_diagnostics.csv"),
        ):
            diag_path = record.latest_run_dir / diag_name
            metrics_path = metrics_root / f"{role}_training_metrics.csv"
            if not diag_path.exists() or not metrics_path.exists():
                continue
            raw = pd.read_csv(diag_path)
            metrics = pd.read_csv(metrics_path)
            if raw.empty or metrics.empty:
                continue
            metrics["num_timesteps"] = pd.to_numeric(metrics[first_existing(metrics, ["num_timesteps", "timesteps", "timestep"])], errors="coerce")
            metrics["outer_iteration"] = metrics["num_timesteps"] / float(n_steps)
            cumulative = metrics["n_updates"].astype(int).tolist()
            start = 0
            out_rows = []
            for idx, stop in enumerate(cumulative):
                stop = int(stop)
                if stop <= start:
                    continue
                chunk = raw.iloc[start:stop].copy()
                start = stop
                if chunk.empty:
                    continue
                actor_series = None
                for col in ("unclipped_actor_surrogate_change", "actual_C_change", "actor_surrogate_change"):
                    if col in chunk.columns:
                        actor_series = pd.to_numeric(chunk[col], errors="coerce")
                        break
                value_series = pd.to_numeric(chunk["value_loss_change"], errors="coerce") if "value_loss_change" in chunk.columns else pd.Series(dtype=float)
                entropy_series = pd.to_numeric(chunk["entropy_change"], errors="coerce") if "entropy_change" in chunk.columns else pd.Series(dtype=float)
                merit_series = pd.to_numeric(chunk["mixed_merit_change"], errors="coerce") if "mixed_merit_change" in chunk.columns else pd.Series(dtype=float)
                out_rows.append(
                    {
                        "method": record.method,
                        "optimizer_role": role,
                        "num_timesteps": float(metrics.iloc[idx]["num_timesteps"]),
                        "outer_iteration": float(metrics.iloc[idx]["outer_iteration"]),
                        "beta_raw": pd.to_numeric(chunk["beta_raw"], errors="coerce").mean() if "beta_raw" in chunk.columns else np.nan,
                        "gamma_raw": pd.to_numeric(chunk["gamma_raw"], errors="coerce").mean() if "gamma_raw" in chunk.columns else np.nan,
                        "beta_eff": pd.to_numeric(chunk["beta_eff"], errors="coerce").mean() if "beta_eff" in chunk.columns else np.nan,
                        "gamma_eff": pd.to_numeric(chunk["gamma_eff"], errors="coerce").mean() if "gamma_eff" in chunk.columns else np.nan,
                        "gamma_active_frac": pd.to_numeric(chunk["gamma_active_frac"], errors="coerce").mean() if "gamma_active_frac" in chunk.columns else np.nan,
                        "fallback_to_noG_frac": pd.to_numeric(chunk["fallback_to_noG"], errors="coerce").mean() if "fallback_to_noG" in chunk.columns else np.nan,
                        "selected_direction": chunk["direction_mode"].dropna().astype(str).mode().iloc[0] if "direction_mode" in chunk.columns and not chunk["direction_mode"].dropna().empty else "",
                        "update_norm_pre_cap": pd.to_numeric(chunk["update_norm_pre_cap"], errors="coerce").mean() if "update_norm_pre_cap" in chunk.columns else np.nan,
                        "update_norm_post_cap": pd.to_numeric(chunk["update_norm_post_cap"], errors="coerce").mean() if "update_norm_post_cap" in chunk.columns else np.nan,
                        "cap_active_frac": pd.to_numeric(chunk["cap_active"], errors="coerce").mean() if "cap_active" in chunk.columns else np.nan,
                        "approx_kl": pd.to_numeric(chunk["approx_kl"], errors="coerce").mean() if "approx_kl" in chunk.columns else np.nan,
                        "clip_fraction": pd.to_numeric(chunk["clip_fraction"], errors="coerce").mean() if "clip_fraction" in chunk.columns else np.nan,
                        "actor_update_norm": pd.to_numeric(chunk["actor_update_norm"], errors="coerce").mean() if "actor_update_norm" in chunk.columns else np.nan,
                        "logstd_update_norm": pd.to_numeric(chunk["logstd_update_norm"], errors="coerce").mean() if "logstd_update_norm" in chunk.columns else np.nan,
                        "critic_update_norm": pd.to_numeric(chunk["critic_update_norm"], errors="coerce").mean() if "critic_update_norm" in chunk.columns else np.nan,
                        "actor_fraction_of_update": pd.to_numeric(chunk["actor_fraction_of_update"], errors="coerce").mean() if "actor_fraction_of_update" in chunk.columns else np.nan,
                        "logstd_fraction_of_update": pd.to_numeric(chunk["logstd_fraction_of_update"], errors="coerce").mean() if "logstd_fraction_of_update" in chunk.columns else np.nan,
                        "critic_fraction_of_update": pd.to_numeric(chunk["critic_fraction_of_update"], errors="coerce").mean() if "critic_fraction_of_update" in chunk.columns else np.nan,
                        "value_loss_change": float(value_series.mean()) if not value_series.empty else np.nan,
                        "actor_surrogate_change": float(actor_series.mean()) if actor_series is not None and not actor_series.empty else np.nan,
                        "entropy_change": float(entropy_series.mean()) if not entropy_series.empty else np.nan,
                        "mixed_merit_change": float(merit_series.mean()) if not merit_series.empty else np.nan,
                    }
                )
            if out_rows:
                frames.append(pd.DataFrame(out_rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_checkpoint_protocol(saved_run: SavedRun, checkpoint_path: pathlib.Path, *, method: str, deterministic: bool, adv_strength: float, operating_mode: Optional[str], device: str, n_eval_episodes: int, protocol: str, timestep: float) -> Dict[str, object]:
    vec_env = make_eval_vec_env(saved_run=saved_run, adv_impact="control", adv_strength=adv_strength, device=device)
    try:
        if checkpoint_path.is_file() and checkpoint_path.suffix == ".zip":
            model = RARL.load(str(saved_run.model_dir), env=vec_env, device=device)
            protagonist_cls = model.protagonist.__class__
            model.protagonist = protagonist_cls.load(str(checkpoint_path), env=vec_env, device=device)
        else:
            model = RARL.load(str(checkpoint_path), env=vec_env, device=device)
        set_rarl_eval_mode(model, vec_env, operating_mode=operating_mode, adv_strength=adv_strength)
        episode_rewards: List[float] = []
        perturb_norms: List[float] = []
        clip_fractions: List[float] = []
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_perturb: List[float] = []
        ep_clip: List[float] = []
        while len(episode_rewards) < n_eval_episodes:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            ep_reward += float(rewards[0])
            info = infos[0]
            ep_perturb.append(float(info.get("applied_control_perturbation_norm", info.get("applied_disturbance_norm", 0.0))))
            ep_clip.append(float(info.get("action_clip_fraction", 0.0)))
            if bool(dones[0]):
                episode_rewards.append(ep_reward)
                perturb_norms.append(float(np.mean(ep_perturb)) if ep_perturb else 0.0)
                clip_fractions.append(float(np.mean(ep_clip)) if ep_clip else 0.0)
                obs = vec_env.reset()
                ep_reward = 0.0
                ep_perturb = []
                ep_clip = []
        return {
            "method": method,
            "eval_type": protocol,
            "timestep": float(timestep),
            "outer_iteration": np.nan,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "applied_perturbation_norm": float(np.mean(perturb_norms)),
            "clip_fraction_eval": float(np.mean(clip_fractions)),
        }
    finally:
        vec_env.close()


def evaluate_stochastic_curves(record: RunRecord, env_id: str, *, device: str, n_eval_episodes: int) -> pd.DataFrame:
    if record.latest_run_dir is None:
        return pd.DataFrame()
    analysis_dir = record.analysis_dir
    inventory_path = analysis_dir / "checkpoint_inventory.csv"
    if not inventory_path.exists():
        return pd.DataFrame()
    checkpoint_df = pd.read_csv(inventory_path).sort_values("timesteps")
    saved_run = build_saved_run(record.latest_run_dir, record.method)
    n_steps = int(saved_run.args_data.get("n_steps", 2048))
    rows: List[Dict[str, object]] = []
    for _, row in checkpoint_df.iterrows():
        checkpoint_path = record.latest_run_dir / str(row["checkpoint_file"])
        timestep = float(row["timesteps"])
        rows.append(
            evaluate_checkpoint_protocol(
                saved_run,
                checkpoint_path,
                method=record.method,
                deterministic=False,
                adv_strength=0.0,
                operating_mode=None,
                device=device,
                n_eval_episodes=n_eval_episodes,
                protocol="clean_stochastic",
                timestep=timestep,
            )
        )
        rows.append(
            evaluate_checkpoint_protocol(
                saved_run,
                checkpoint_path,
                method=record.method,
                deterministic=False,
                adv_strength=1.0,
                operating_mode="protagonist",
                device=device,
                n_eval_episodes=n_eval_episodes,
                protocol="control_adv_stochastic",
                timestep=timestep,
            )
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["outer_iteration"] = df["timestep"] / float(n_steps)
    return df


def evaluate_final_robustness(record: RunRecord, env_id: str, strengths: Sequence[float], *, device: str, n_eval_episodes: int) -> pd.DataFrame:
    if record.latest_run_dir is None:
        return pd.DataFrame()
    saved_run = build_saved_run(record.latest_run_dir, record.method)
    rows: List[Dict[str, object]] = []
    for strength in strengths:
        rows.append(
            evaluate_checkpoint_protocol(
                saved_run,
                record.latest_run_dir,
                method=record.method,
                deterministic=False,
                adv_strength=float(strength),
                operating_mode="protagonist" if strength > 0 else None,
                device=device,
                n_eval_episodes=n_eval_episodes,
                protocol="robustness_sweep",
                timestep=float("nan"),
            )
            | {"adv_strength": float(strength)}
        )
    return pd.DataFrame(rows)


def build_summary(training_df: pd.DataFrame, eval_df: pd.DataFrame, robustness_df: pd.DataFrame, records: Sequence[RunRecord]) -> pd.DataFrame:
    rows = []
    run_summaries: Dict[str, Dict[str, object]] = {}
    for record in records:
        if (record.analysis_dir / "run_summary.csv").exists():
            run_summaries[record.method] = pd.read_csv(record.analysis_dir / "run_summary.csv").iloc[0].to_dict()
        else:
            run_summaries[record.method] = {"status": record.status, "error_message": record.error_message}
    for method, run_summary in run_summaries.items():
        train_stats = aggregate_series(training_df[training_df["method"] == method], "episode_return")
        clean_det_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "clean_deterministic")], "mean_reward")
        clean_stoch_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "clean_stochastic")], "mean_reward")
        adv_det_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "control_adv_deterministic")], "mean_reward")
        adv_stoch_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "control_adv_stochastic")], "mean_reward")
        robust_sub = robustness_df[robustness_df["method"] == method].sort_values("adv_strength")
        robust_auc = compute_auc(robust_sub["adv_strength"], robust_sub["mean_reward"]) if not robust_sub.empty else float("nan")
        row = dict(run_summary)
        row.update(
            {
                "method": method,
                "status": run_summary.get("status", "completed"),
                "final_training_return": train_stats["final"],
                "last5_training_mean": train_stats["last5_mean"],
                "training_auc": train_stats["auc"],
                "final_clean_deterministic": clean_det_stats["final"],
                "last5_clean_deterministic_mean": clean_det_stats["last5_mean"],
                "final_clean_stochastic": clean_stoch_stats["final"],
                "last5_clean_stochastic_mean": clean_stoch_stats["last5_mean"],
                "final_control_adv_deterministic": adv_det_stats["final"],
                "last5_control_adv_deterministic_mean": adv_det_stats["last5_mean"],
                "final_control_adv_stochastic": adv_stoch_stats["final"],
                "last5_control_adv_stochastic_mean": adv_stoch_stats["last5_mean"],
                "clean_auc": clean_stoch_stats["auc"],
                "control_adv_auc": adv_stoch_stats["auc"],
                "robustness_auc": robust_auc,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_training(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    for method, group in df.groupby("method"):
        group = group.sort_values("outer_iteration")
        smoothed = moving_average(group["episode_return"], window=25)
        ax.plot(group["outer_iteration"], smoothed, label=method, color=colors.get(method))
    ax.set_title("Training return (moving average)")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)


def plot_band(ax, df: pd.DataFrame, eval_type: str, title: str, colors: Dict[str, str]) -> None:
    sub = df[df["eval_type"] == eval_type].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        mean = pd.to_numeric(group["mean_reward"], errors="coerce")
        std = pd.to_numeric(group["std_reward"], errors="coerce").fillna(0.0)
        mean_smoothed = moving_average(mean, window=3)
        std_smoothed = moving_average(std, window=3)
        ax.plot(group["outer_iteration"], mean_smoothed, label=method, color=colors.get(method))
        ax.fill_between(group["outer_iteration"], mean_smoothed - std_smoothed, mean_smoothed + std_smoothed, alpha=0.15, color=colors.get(method))
    ax.set_title(f"{title} (moving average)")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)


def plot_qp_panel(axs, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        for ax in axs.flat:
            ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        return
    sub = df[df["optimizer_role"] == "protagonist"].copy()
    specs = [
        ("beta_raw", "Beta raw"),
        ("gamma_raw", "Gamma raw"),
        ("fallback_to_noG_frac", "Fallback frac"),
        ("gamma_active_frac", "Gamma active frac"),
    ]
    for ax, (col, title) in zip(axs.flat, specs):
        for method, group in sub.groupby("method"):
            group = group.sort_values("outer_iteration")
            ax.plot(group["outer_iteration"], pd.to_numeric(group[col], errors="coerce"), color=colors.get(method), label=method)
        ax.set_title(title)
        ax.set_xlabel("Outer iteration")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)


def plot_block_update_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    sub = df[df["optimizer_role"] == "protagonist"].copy()
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["actor_update_norm"], errors="coerce"), color=colors.get(method), label=f"{method} actor")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["logstd_update_norm"], errors="coerce"), color=colors.get(method), linestyle=":", label=f"{method} logstd")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["critic_update_norm"], errors="coerce"), color=colors.get(method), linestyle="--", label=f"{method} critic")
    ax.set_title("Block update norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Norm")
    ax.grid(alpha=0.3)


def plot_param_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    sub = df[df["agent_name"] == "protagonist"].copy()
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["actor_param_norm"], errors="coerce"), color=colors.get(method), label=f"{method} actor")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["log_std_norm"], errors="coerce"), color=colors.get(method), linestyle=":", label=f"{method} logstd")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["critic_param_norm"], errors="coerce"), color=colors.get(method), linestyle="--", label=f"{method} critic")
    ax.set_title("Protagonist parameter norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)


def plot_merit_components(axs, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        for ax in axs.flat:
            ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        return
    sub = df[df["optimizer_role"] == "protagonist"].copy()
    specs = [
        ("mixed_merit_change", "Mixed merit change"),
        ("actor_surrogate_change", "Actor surrogate change"),
        ("value_loss_change", "Value loss change"),
        ("entropy_change", "Entropy change"),
    ]
    for ax, (col, title) in zip(axs.flat, specs):
        for method, group in sub.groupby("method"):
            group = group.sort_values("outer_iteration")
            ax.plot(group["outer_iteration"], pd.to_numeric(group[col], errors="coerce"), color=colors.get(method), label=method)
        ax.set_title(title)
        ax.set_xlabel("Outer iteration")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)


def plot_robustness(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    for method, group in df.groupby("method"):
        group = group.sort_values("adv_strength")
        ax.plot(group["adv_strength"], pd.to_numeric(group["mean_reward"], errors="coerce"), marker="o", color=colors.get(method), label=method)
    ax.set_title("Robustness sweep")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)


def plot_final_bar(ax, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    order = summary_df["method"].tolist()
    x = np.arange(len(order))
    width = 0.18
    ax.bar(x - 1.5 * width, summary_df["final_clean_stochastic"], width=width, label="clean_stoch")
    ax.bar(x - 0.5 * width, summary_df["final_clean_deterministic"], width=width, label="clean_det")
    ax.bar(x + 0.5 * width, summary_df["final_control_adv_stochastic"], width=width, label="adv_stoch")
    ax.bar(x + 1.5 * width, summary_df["robustness_auc"], width=width, label="robust_auc")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Final comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_convergence_slopes(ax, df: pd.DataFrame) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    pivot = df.pivot(index="method", columns="metric", values="slope_last5")
    metrics = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(metrics), 1)
    for idx, metric in enumerate(metrics):
        ax.bar(x - 0.4 + (idx + 0.5) * width, pivot[metric].to_numpy(dtype=float), width=width, label=metric)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=20, ha="right")
    ax.set_title("Last-window convergence slopes")
    ax.set_ylabel("Slope")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def make_collage(plot_paths: Sequence[pathlib.Path], output_path: pathlib.Path, cols: int = 2) -> None:
    valid_paths: List[pathlib.Path] = []
    images: List[Image.Image] = []
    for path in plot_paths:
        if not path.exists():
            continue
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except (UnidentifiedImageError, OSError):
            continue
    if not images:
        return
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = math.ceil(len(images) / cols)
    pad = 20
    header_h = 36
    canvas = Image.new("RGB", (pad + cols * (cell_w + pad), pad + rows * (cell_h + header_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for idx, (img, path) in enumerate(zip(images, valid_paths)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + header_h + pad)
        draw.text((x0 + 8, y0 + 8), f"{idx + 1}. {path.name}", fill="black", font=font)
        thumb = img.copy()
        thumb.thumbnail((cell_w, cell_h))
        canvas.paste(thumb, (x0 + (cell_w - thumb.width) // 2, y0 + header_h + (cell_h - thumb.height) // 2))
    canvas.save(output_path)


def build_convergence_diagnostics(training_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metric_map = {
        "training_return": (training_df, "episode_return"),
        "clean_stochastic": (eval_df[eval_df["eval_type"] == "clean_stochastic"], "mean_reward"),
        "control_adv_stochastic": (eval_df[eval_df["eval_type"] == "control_adv_stochastic"], "mean_reward"),
        "clean_deterministic": (eval_df[eval_df["eval_type"] == "clean_deterministic"], "mean_reward"),
    }
    methods = sorted(set(training_df["method"].tolist()) | set(eval_df["method"].tolist()))
    for method in methods:
        for metric_name, (frame, value_col) in metric_map.items():
            sub = frame[frame["method"] == method]
            stats = aggregate_series(sub, value_col)
            prev5 = stats["prev5_mean"]
            last5 = stats["last5_mean"]
            slope = stats["slope_last5"]
            delta = last5 - prev5 if np.isfinite(last5) and np.isfinite(prev5) else np.nan
            threshold = max(5.0, 0.03 * max(abs(prev5) if np.isfinite(prev5) else 0.0, 10.0))
            clearly_rising = bool(np.isfinite(delta) and np.isfinite(slope) and delta > threshold and slope > 0.0)
            rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "final": stats["final"],
                    "last5_mean": last5,
                    "prev5_mean": prev5,
                    "delta_last5_vs_prev5": delta,
                    "slope_last5": slope,
                    "clearly_rising": clearly_rising,
                }
            )
    return pd.DataFrame(rows)


def should_continue(convergence_df: pd.DataFrame) -> bool:
    if convergence_df.empty:
        return False
    per_method = (
        convergence_df.groupby("method")["clearly_rising"]
        .sum()
        .reset_index(name="num_rising_metrics")
    )
    return int((per_method["num_rising_metrics"] >= 2).sum()) >= 2


def write_reports(
    output_root: pathlib.Path,
    budget_audit_root: pathlib.Path,
    final_budget: int,
    config_audit: pd.DataFrame,
    summary_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    qp_diag_df: pd.DataFrame,
) -> None:
    budget_audit_df = pd.read_csv(budget_audit_root / "full_convergence_seed0_original_budget_audit.csv")
    outer_n = int(budget_audit_df.iloc[0]["n_iter_outer"])
    plateau_text = "still rising" if should_continue(convergence_df) else "roughly plateau / not clearly rising enough for another uniform extension"
    best_final_clean = summary_df.sort_values("final_clean_deterministic", ascending=False).iloc[0]
    best_last5_clean = summary_df.sort_values("last5_clean_deterministic_mean", ascending=False).iloc[0]
    best_adv_det = summary_df.sort_values("final_control_adv_deterministic", ascending=False).iloc[0]
    proposed_row = summary_df[summary_df["method"] == "proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG"].iloc[0]
    sgd_row = summary_df[summary_df["method"] == "sgd_lr1e-3"].iloc[0]
    egm_row = summary_df[summary_df["method"] == "egm_lr5e-3"].iloc[0]
    ppm_row = summary_df[summary_df["method"] == "ppm_lr5e-3"].iloc[0]

    lines = [
        "# Full Convergence Seed0 Report",
        "",
        f"1. 原始 RARL HalfCheetah 默认 outer iterations 是 `{outer_n}`。",
        f"2. 本实验最终用了 `{final_budget}` outer iterations。",
        f"3. 四条曲线整体判断：`{plateau_text}`。",
        f"4. proposed(training return) = `{proposed_row['final_training_return']:.6g}`; PPM/EGM/SGD = `{ppm_row['final_training_return']:.6g}` / `{egm_row['final_training_return']:.6g}` / `{sgd_row['final_training_return']:.6g}`。",
        f"5. proposed(clean stochastic) = `{proposed_row['final_clean_stochastic']:.6g}`; PPM/EGM/SGD = `{ppm_row['final_clean_stochastic']:.6g}` / `{egm_row['final_clean_stochastic']:.6g}` / `{sgd_row['final_clean_stochastic']:.6g}`。",
        f"6. proposed(control-adv stochastic) = `{proposed_row['final_control_adv_stochastic']:.6g}`; PPM/EGM/SGD = `{ppm_row['final_control_adv_stochastic']:.6g}` / `{egm_row['final_control_adv_stochastic']:.6g}` / `{sgd_row['final_control_adv_stochastic']:.6g}`。",
        f"7. deterministic eval: best final clean deterministic is `{best_final_clean['method']}` at `{best_final_clean['final_clean_deterministic']:.6g}`; best last5 clean deterministic is `{best_last5_clean['method']}` at `{best_last5_clean['last5_clean_deterministic_mean']:.6g}`.",
        f"8. PPM/EGM lr=5e-3 stability: PPM status=`{ppm_row.get('status', 'completed')}`, EGM status=`{egm_row.get('status', 'completed')}`.",
        f"9. proposed gamma active mean = `{float(qp_diag_df['gamma_active_frac'].mean()):.6g}`." if not qp_diag_df.empty else "9. proposed gamma diagnostics missing.",
        f"10. fallback mean = `{float(qp_diag_df['fallback_to_noG_frac'].mean()):.6g}`." if not qp_diag_df.empty else "10. fallback diagnostics missing.",
        f"11. actor/log_std/critic all updated: protagonist means `{float(qp_diag_df['actor_update_norm'].mean()):.3e}` / `{float(qp_diag_df['logstd_update_norm'].mean()):.3e}` / `{float(qp_diag_df['critic_update_norm'].mean()):.3e}`." if not qp_diag_df.empty else "11. block-update diagnostics missing.",
        f"12. 论文主图候选：`{best_last5_clean['method']}` by last5 clean deterministic, and compare with `{best_adv_det['method']}` on control-adv deterministic if they differ.",
        "",
        "## Readout",
        "",
        f"- Best proposed by final clean deterministic: `{best_final_clean['method']}` = `{best_final_clean['final_clean_deterministic']:.6g}`",
        f"- Best proposed by last5 clean deterministic: `{best_last5_clean['method']}` = `{best_last5_clean['last5_clean_deterministic_mean']:.6g}`",
        f"- Best method by control-adv deterministic: `{best_adv_det['method']}` = `{best_adv_det['final_control_adv_deterministic']:.6g}`",
    ]
    if proposed_row["final_training_return"] > max(ppm_row["final_training_return"], egm_row["final_training_return"], sgd_row["final_training_return"]):
        lines.extend([
            "",
            "**adaptive full-param performance-merit step-size is effective.**",
        ])
    if proposed_row["final_clean_deterministic"] <= max(ppm_row["final_clean_deterministic"], egm_row["final_clean_deterministic"], sgd_row["final_clean_deterministic"]):
        lines.extend([
            "",
            "**current second-direction QP does not yet clearly dominate all baselines under the full-budget online screen.**",
        ])
    (output_root / "full_convergence_seed0_report.md").write_text("\n".join(lines), encoding="utf-8")

    metric_lines = [
        "# Full Convergence Metric Report",
        "",
        "```csv",
        summary_df.to_csv(index=False),
        "```",
    ]
    (output_root / "full_convergence_seed0_metric_report.md").write_text("\n".join(metric_lines), encoding="utf-8")

    qp_lines = [
        "# Full Convergence QP Diagnostics Report",
        "",
        "Rows below are protagonist/adversary aggregated diagnostics recovered from raw optimizer logs.",
        "",
        "No QP diagnostics recovered." if qp_diag_df.empty else "```csv\n" + qp_diag_df.to_csv(index=False) + "```",
    ]
    (output_root / "full_convergence_seed0_qp_diagnostics_report.md").write_text("\n".join(qp_lines), encoding="utf-8")

    conv_lines = [
        "# Full Convergence Continuation Report",
        "",
        "- Continuation heuristic: a metric is marked `clearly_rising` when last5 mean exceeds previous5 mean by more than `max(5, 3%)` and the last-window slope is positive.",
        "- A budget extension is triggered when at least two methods each have at least two primary metrics marked `clearly_rising`.",
        "",
        "```csv",
        convergence_df.to_csv(index=False),
        "```",
    ]
    (output_root / "full_convergence_seed0_convergence_report.md").write_text("\n".join(conv_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    config_rows = []
    strengths = [float(token) for token in args.robustness_strengths.split(",") if token.strip()]
    for spec in candidates:
        config_rows.append(
            {
                "method": spec.method,
                "optimizer": spec.optimizer,
                "lr": spec.lr,
                "max_grad_norm": spec.max_grad_norm,
                "vf_coef": spec.vf_coef,
                "optimizer_kwargs": repr(spec.optimizer_kwargs),
            }
        )
    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(output_root / "full_convergence_seed0_config_audit.csv", index=False)

    budgets_to_try = [args.initial_iterations, 150, 200]
    final_budget = args.initial_iterations
    final_records: List[RunRecord] = []
    final_training_df = pd.DataFrame()
    final_eval_df = pd.DataFrame()
    final_param_df = pd.DataFrame()
    final_update_df = pd.DataFrame()
    final_qp_diag_df = pd.DataFrame()
    final_robustness_df = pd.DataFrame()
    final_summary_df = pd.DataFrame()
    final_convergence_df = pd.DataFrame()

    for budget_idx, budget in enumerate(budgets_to_try):
        budget_root = output_root / f"budget_{budget}"
        budget_root.mkdir(parents=True, exist_ok=True)
        records = [ensure_run_for_budget(spec, args, budget_root, budget) for spec in candidates]

        training_frames: List[pd.DataFrame] = []
        eval_frames: List[pd.DataFrame] = []
        param_frames: List[pd.DataFrame] = []
        stochastic_frames: List[pd.DataFrame] = []
        robustness_frames: List[pd.DataFrame] = []

        for record in records:
            if record.latest_run_dir is None or not record.analysis_dir.exists() or not (record.analysis_dir / "run_summary.csv").exists():
                continue
            args_path = next(path for path in record.latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists()) / "args.yml"
            args_data = load_yaml(args_path)
            n_steps = int(args_data.get("n_steps", 2048))
            training_frames.append(read_training_frame(record.run_root, record.latest_run_dir, record.method))
            eval_frames.append(read_eval_frame(record.analysis_dir / "clean_eval_returns.csv", record.method, n_steps, "clean_deterministic"))
            eval_frames.append(read_eval_frame(record.analysis_dir / "adversarial_eval_returns.csv", record.method, n_steps, "control_adv_deterministic"))
            param_frames.append(read_param_frame(record.run_root, record.latest_run_dir, record.method))
            stochastic_df = evaluate_stochastic_curves(record, args.env, device=args.device, n_eval_episodes=args.n_eval_episodes)
            if not stochastic_df.empty:
                stochastic_frames.append(stochastic_df)
            robust_df = evaluate_final_robustness(record, args.env, strengths, device=args.device, n_eval_episodes=args.n_eval_episodes)
            if not robust_df.empty:
                robustness_frames.append(robust_df)

        training_df = pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame(columns=["method", "outer_iteration", "episode_return"])
        eval_df = pd.concat(eval_frames + stochastic_frames, ignore_index=True) if (eval_frames or stochastic_frames) else pd.DataFrame(columns=["method", "eval_type", "outer_iteration", "mean_reward", "std_reward"])
        param_df = pd.concat(param_frames, ignore_index=True) if param_frames else pd.DataFrame()
        update_df = recover_update_frames(records)
        qp_diag_df = recover_qp_diagnostics(records)
        robustness_df = pd.concat(robustness_frames, ignore_index=True) if robustness_frames else pd.DataFrame(columns=["method", "adv_strength", "mean_reward"])
        summary_df = build_summary(training_df, eval_df, robustness_df, records)
        convergence_df = build_convergence_diagnostics(training_df, eval_df)

        final_budget = budget
        final_records = records
        final_training_df = training_df
        final_eval_df = eval_df
        final_param_df = param_df
        final_update_df = update_df
        final_qp_diag_df = qp_diag_df
        final_robustness_df = robustness_df
        final_summary_df = summary_df
        final_convergence_df = convergence_df

        if budget_idx == len(budgets_to_try) - 1:
            break
        if should_continue(convergence_df):
            continue
        break

    final_training_df.to_csv(output_root / "full_convergence_seed0_training_curves.csv", index=False)
    final_eval_df.to_csv(output_root / "full_convergence_seed0_eval_curves.csv", index=False)
    final_robustness_df.to_csv(output_root / "full_convergence_seed0_robustness_sweep.csv", index=False)
    final_param_df.to_csv(output_root / "full_convergence_seed0_param_norms.csv", index=False)
    final_update_df.to_csv(output_root / "full_convergence_seed0_update_norms.csv", index=False)
    final_qp_diag_df.to_csv(output_root / "full_convergence_seed0_qp_diagnostics.csv", index=False)
    final_convergence_df.to_csv(output_root / "full_convergence_seed0_convergence_diagnostics.csv", index=False)
    final_summary_df.to_csv(output_root / "full_convergence_seed0_summary.csv", index=False)

    colors = {
        "proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG": "tab:blue",
        "ppm_lr5e-3": "tab:red",
        "egm_lr5e-3": "tab:green",
        "sgd_lr1e-3": "tab:gray",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_training(ax, final_training_df, colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_training_return.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for eval_type, filename, title in [
        ("clean_stochastic", "full_clean_stochastic.png", "Clean stochastic"),
        ("clean_deterministic", "full_clean_deterministic.png", "Clean deterministic"),
        ("control_adv_stochastic", "full_control_adv_stochastic.png", "Control-adv stochastic"),
        ("control_adv_deterministic", "full_control_adv_deterministic.png", "Control-adv deterministic"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_band(ax, final_eval_df, eval_type, title, colors)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_robustness(ax, final_robustness_df, colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_robustness_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_qp_panel(axs, final_qp_diag_df, colors)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_qp_beta_gamma_fallback.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_block_update_norms(ax, final_update_df, colors)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_update_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_param_norms(ax, final_param_df, colors)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_param_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_merit_components(axs, final_qp_diag_df, colors)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_merit_components.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_final_bar(ax, final_summary_df)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_final_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_convergence_slopes(ax, final_convergence_df)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_convergence_slope.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    collage_paths = [
        plots_dir / "full_training_return.png",
        plots_dir / "full_clean_stochastic.png",
        plots_dir / "full_clean_deterministic.png",
        plots_dir / "full_control_adv_stochastic.png",
        plots_dir / "full_control_adv_deterministic.png",
        plots_dir / "full_robustness_sweep.png",
        plots_dir / "full_qp_beta_gamma_fallback.png",
        plots_dir / "full_merit_components.png",
        plots_dir / "full_update_norms.png",
        plots_dir / "full_param_norms.png",
        plots_dir / "full_final_bar.png",
        plots_dir / "full_convergence_slope.png",
    ]
    make_collage(collage_paths, plots_dir / "full_all_plots_big.png", cols=2)

    write_reports(
        output_root,
        output_root,
        final_budget,
        config_df,
        final_summary_df,
        final_convergence_df,
        final_qp_diag_df,
    )


if __name__ == "__main__":
    main()
