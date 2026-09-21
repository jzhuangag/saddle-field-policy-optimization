from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.RARL import RARL
from scripts.full_policy_followup_common import SavedRun, load_yaml, make_eval_vec_env, set_rarl_eval_mode
from scripts.run_perflyap_stage412_actual_merit_oracle import rollout_return_with_model


@dataclass(frozen=True)
class Candidate:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]
    cost_mode: str
    selector_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 5 perfLyap short online screen")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--short-horizon", type=int, default=16)
    parser.add_argument("--short-return-episodes", type=int, default=1)
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
        if isinstance(value, float) and not np.isfinite(value):
            continue
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


def pick_column(frame: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise KeyError(f"Missing required columns. Tried {list(candidates)} but only found {list(frame.columns)}")
    return None


def read_method_frames(method_label: str, latest_run_dir: pathlib.Path, analysis_dir: pathlib.Path) -> Dict[str, pd.DataFrame]:
    config_dir = next(path for path in latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists())
    args_data = load_yaml(config_dir / "args.yml")
    n_steps = int(args_data.get("n_steps", 2048))

    training = pd.read_csv(analysis_dir / "training_episode_returns.csv")
    train_time_col = pick_column(training, ["timesteps", "timestep", "total_timesteps", "cumulative_timesteps"])
    train_return_col = pick_column(training, ["episode_return", "reward", "ep_rew_mean", "return"])
    training["timestep"] = pd.to_numeric(training[train_time_col], errors="coerce")
    training["episode_return"] = pd.to_numeric(training[train_return_col], errors="coerce")
    training["method"] = method_label
    training["outer_iteration"] = training["timestep"] / float(n_steps)

    clean = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
    clean_time_col = pick_column(clean, ["timesteps", "timestep", "total_timesteps"])
    clean_mean_col = pick_column(clean, ["mean_reward", "clean_mean", "eval_mean_reward"])
    clean_std_col = pick_column(clean, ["std_reward", "clean_std", "eval_std_reward"], required=False)
    clean["timestep"] = pd.to_numeric(clean[clean_time_col], errors="coerce")
    clean["mean_reward"] = pd.to_numeric(clean[clean_mean_col], errors="coerce")
    clean["std_reward"] = pd.to_numeric(clean[clean_std_col], errors="coerce") if clean_std_col else np.nan
    clean["method"] = method_label
    clean["outer_iteration"] = clean["timestep"] / float(n_steps)

    adv = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
    adv_time_col = pick_column(adv, ["timesteps", "timestep", "total_timesteps"])
    adv_mean_col = pick_column(adv, ["mean_reward", "adv_mean", "adversarial_mean", "control_adv_mean"])
    adv_std_col = pick_column(adv, ["std_reward", "adv_std", "adversarial_std", "control_adv_std"], required=False)
    adv["timestep"] = pd.to_numeric(adv[adv_time_col], errors="coerce")
    adv["mean_reward"] = pd.to_numeric(adv[adv_mean_col], errors="coerce")
    adv["std_reward"] = pd.to_numeric(adv[adv_std_col], errors="coerce") if adv_std_col else np.nan
    adv["method"] = method_label
    adv["outer_iteration"] = adv["timestep"] / float(n_steps)

    param = pd.read_csv(analysis_dir / "parameter_norms.csv")
    param_time_col = pick_column(param, ["num_timesteps", "timesteps", "timestep"])
    param["num_timesteps"] = pd.to_numeric(param[param_time_col], errors="coerce")
    param["actor_param_norm"] = pd.to_numeric(param[pick_column(param, ["actor_param_norm", "actor_norm"])], errors="coerce")
    param["critic_param_norm"] = pd.to_numeric(param[pick_column(param, ["critic_param_norm", "critic_norm"])], errors="coerce")
    param["log_std_norm"] = pd.to_numeric(param[pick_column(param, ["log_std_norm", "logstd_norm"])], errors="coerce")
    param["total_param_norm"] = pd.to_numeric(param[pick_column(param, ["total_param_norm", "total_norm"])], errors="coerce")
    param["method"] = method_label
    param["outer_iteration"] = param["num_timesteps"] / float(n_steps)

    empty_metric_cols = [
        "method",
        "optimizer_role",
        "num_timesteps",
        "outer_iteration",
        "actor_update_norm",
        "critic_update_norm",
        "logstd_update_norm",
        "approx_kl",
        "clip_fraction",
        "lr",
        "max_grad_norm",
        "vf_coef",
        "n_updates",
    ]
    protagonist_metrics_path = analysis_dir / "protagonist_training_metrics.csv"
    if protagonist_metrics_path.exists():
        protagonist_metrics = pd.read_csv(protagonist_metrics_path)
        protagonist_metrics["num_timesteps"] = pd.to_numeric(
            protagonist_metrics[pick_column(protagonist_metrics, ["num_timesteps", "timesteps", "timestep"])], errors="coerce"
        )
        protagonist_metrics["method"] = method_label
        protagonist_metrics["optimizer_role"] = "protagonist"
        protagonist_metrics["outer_iteration"] = protagonist_metrics["num_timesteps"] / float(n_steps)
    else:
        protagonist_metrics = pd.DataFrame(columns=empty_metric_cols)

    adversary_metrics_path = analysis_dir / "adversary_training_metrics.csv"
    if adversary_metrics_path.exists():
        adversary_metrics = pd.read_csv(adversary_metrics_path)
        adversary_metrics["num_timesteps"] = pd.to_numeric(
            adversary_metrics[pick_column(adversary_metrics, ["num_timesteps", "timesteps", "timestep"])], errors="coerce"
        )
        adversary_metrics["method"] = method_label
        adversary_metrics["optimizer_role"] = "adversary"
        adversary_metrics["outer_iteration"] = adversary_metrics["num_timesteps"] / float(n_steps)
    else:
        adversary_metrics = pd.DataFrame(columns=empty_metric_cols)

    diagnostics = pd.DataFrame()
    for diag_path in sorted(latest_run_dir.glob("*diagnostics.csv")):
        diag = pd.read_csv(diag_path)
        role = "protagonist" if "protagonist" in diag_path.name else "adversary"
        metrics_df = protagonist_metrics if role == "protagonist" else adversary_metrics
        if metrics_df.empty:
            continue
        cumulative = metrics_df["n_updates"].astype(int).tolist()
        start = 0
        chunks = []
        for idx, stop in enumerate(cumulative):
            stop = int(stop)
            if stop <= start:
                continue
            chunk = diag.iloc[start:stop].copy()
            chunk["num_timesteps"] = float(metrics_df.iloc[idx]["num_timesteps"])
            chunk["outer_iteration"] = float(metrics_df.iloc[idx]["outer_iteration"])
            chunk["optimizer_role"] = role
            chunks.append(chunk)
            start = stop
        if chunks:
            role_df = pd.concat(chunks, ignore_index=True)
            role_df["method"] = method_label
            diagnostics = pd.concat([diagnostics, role_df], ignore_index=True)

    return {
        "training": training,
        "clean": clean,
        "adv": adv,
        "param": param,
        "protagonist_metrics": protagonist_metrics,
        "adversary_metrics": adversary_metrics,
        "diagnostics": diagnostics,
    }


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


def ensure_run(candidate: Candidate, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
    run_root = runs_dir / candidate.method
    saved_models_dir = run_root / "saved_models"
    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    analysis_dir = run_root / "analysis"
    run_root.mkdir(parents=True, exist_ok=True)

    env_root = saved_models_dir / "rarl-ppo" / args.env
    if env_root.exists():
        latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
        if not (analysis_dir / "run_summary.csv").exists():
            analyze_run(pathlib.Path(args.repo_dir), args.python_path, latest_run_dir, analysis_dir, candidate.method)
        return latest_run_dir

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
        str(args.iterations),
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
    return latest_run_dir


def evaluate_checkpoint_protocol(saved_run: SavedRun, checkpoint_path: pathlib.Path, *, method: str, deterministic: bool, adv_strength: float, operating_mode: str | None, device: str, n_eval_episodes: int, protocol: str, timestep: float) -> Dict[str, object]:
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


def evaluate_stochastic_curves(run_root: pathlib.Path, method: str, *, device: str, n_eval_episodes: int) -> pd.DataFrame:
    analysis_dir = run_root / "analysis"
    checkpoint_df = pd.read_csv(analysis_dir / "checkpoint_inventory.csv").sort_values("timesteps")
    latest_run_dir = find_latest_run_dir(run_root / "saved_models", "HalfCheetah-v4")
    saved_run = build_saved_run(latest_run_dir, method)
    n_steps = int(saved_run.args_data.get("n_steps", 2048))
    rows: List[Dict[str, object]] = []
    for _, row in checkpoint_df.iterrows():
        checkpoint_path = latest_run_dir / str(row["checkpoint_file"])
        timestep = float(row["timesteps"])
        rows.append(
            evaluate_checkpoint_protocol(
                saved_run,
                checkpoint_path,
                method=method,
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
                method=method,
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
    df["outer_iteration"] = df["timestep"] / float(n_steps)
    return df


def final_short_return_audits(run_root: pathlib.Path, method: str, *, env_id: str, device: str, horizon: int, episodes: int, seed: int) -> Dict[str, object]:
    latest_run_dir = find_latest_run_dir(run_root / "saved_models", env_id)
    saved_run = build_saved_run(latest_run_dir, method)
    vec_env = make_eval_vec_env(saved_run=saved_run, adv_impact="control", adv_strength=0.0, device=device)
    try:
        model = RARL.load(str(saved_run.model_dir), env=vec_env, device=device)
        clean_mean, _ = rollout_return_with_model(model, env_id=env_id, adv_strength=0.0, episodes=episodes, horizon=horizon, base_seed=seed, control_adv=False)
        adv_mean, _ = rollout_return_with_model(model, env_id=env_id, adv_strength=1.0, episodes=episodes, horizon=horizon, base_seed=seed, control_adv=True)
        return {
            "method": method,
            "short_clean_return_cost": float(-clean_mean),
            "short_rarl_return_cost": float(-0.5 * clean_mean - 0.5 * adv_mean),
        }
    finally:
        vec_env.close()


def load_baseline_method(run_root: pathlib.Path, method: str) -> Dict[str, object]:
    latest_run_dir = find_latest_run_dir(run_root / "saved_models", "HalfCheetah-v4")
    analysis_dir = run_root / "analysis"
    frames = read_method_frames(method, latest_run_dir, analysis_dir)
    summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    return {
        "run_root": run_root,
        "latest_run_dir": latest_run_dir,
        "frames": frames,
        "summary": summary,
    }


def plot_band(ax, df: pd.DataFrame, eval_type: str, title: str, colors: Dict[str, str]) -> None:
    sub = df[df["eval_type"] == eval_type].copy()
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["mean_reward"], label=method, color=colors.get(method))
        ax.fill_between(group["outer_iteration"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], color=colors.get(method), alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)


def plot_training(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["episode_return"], label=method, color=colors.get(method))
    ax.set_title("Training return")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)


def plot_qp_metric(ax, df: pd.DataFrame, y_col: str, title: str, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
        if group.empty or y_col not in group.columns:
            continue
        ax.plot(group["outer_iteration"], pd.to_numeric(group[y_col], errors="coerce"), label=method, color=colors.get(method))
    ax.set_title(title)
    ax.set_xlabel("Outer iteration")
    ax.grid(alpha=0.3)


def plot_update_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["actor_update_norm"], errors="coerce"), label=f"{method} actor", color=colors.get(method))
        ax.plot(group["outer_iteration"], pd.to_numeric(group["critic_update_norm"], errors="coerce"), linestyle="--", color=colors.get(method), label=f"{method} critic")
    ax.set_title("Update norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Norm")
    ax.grid(alpha=0.3)


def plot_param_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    sub = df[df["agent_name"] == "protagonist"].copy()
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["actor_param_norm"], color=colors.get(method), label=f"{method} actor")
        ax.plot(group["outer_iteration"], group["critic_param_norm"], color=colors.get(method), linestyle="--", label=f"{method} critic")
    ax.set_title("Protagonist parameter norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)


def make_collage(plot_paths: Sequence[pathlib.Path], output_path: pathlib.Path, cols: int = 2) -> None:
    images = [Image.open(path).convert("RGB") for path in plot_paths if path.exists()]
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
    existing_paths = [p for p in plot_paths if p.exists()]
    for idx, (img, path) in enumerate(zip(images, existing_paths)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + header_h + pad)
        draw.text((x0 + 8, y0 + 8), f"{idx + 1}. {path.name}", fill="black", font=font)
        thumb = img.copy()
        thumb.thumbnail((cell_w, cell_h))
        canvas.paste(thumb, (x0 + (cell_w - thumb.width) // 2, y0 + header_h + (cell_h - thumb.height) // 2))
    canvas.save(output_path)


def build_candidates() -> List[Candidate]:
    common = {
        "perflyap_scope": "actor_mean_only",
        "lambda_N": 0.0,
        "lambda_P": 1.0,
        "lambda_critic": 0.0,
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
        "selector_beta_grid": "0,0.009,0.012,0.015",
        "selector_gamma_grid": "0,1.2e-05,2.4e-05,3e-05",
        "ls_fit_variant": "LS_all_grid",
    }
    return [
        Candidate(
            "proposed_noG_actor_surrogate",
            "proposed_noG_perfLyap",
            1.0,
            1.0,
            1.0,
            {**common, "cost_mode": "actor_surrogate_cost", "selector_mode": "fixed_nog", "fixed_beta_raw": 0.015, "fixed_gamma_raw": 0.0},
            "actor_surrogate_cost",
            "fixed_nog",
        ),
        Candidate(
            "proposed_qp_actor_surrogate_actual_selector",
            "proposed_qp_perfLyap",
            1.0,
            1.0,
            1.0,
            {**common, "cost_mode": "actor_surrogate_cost", "selector_mode": "actual_surrogate_selector", "fixed_beta_raw": 0.009, "fixed_gamma_raw": 3e-05},
            "actor_surrogate_cost",
            "actual_surrogate_selector",
        ),
        Candidate(
            "proposed_qp_actor_surrogate_LS_capaware",
            "proposed_qp_perfLyap",
            1.0,
            1.0,
            1.0,
            {**common, "cost_mode": "actor_surrogate_cost", "selector_mode": "ls_capaware", "fixed_beta_raw": 0.012, "fixed_gamma_raw": 2.4e-05},
            "actor_surrogate_cost",
            "ls_capaware",
        ),
        Candidate(
            "proposed_noG_unclipped_surrogate",
            "proposed_noG_perfLyap",
            1.0,
            1.0,
            1.0,
            {**common, "cost_mode": "unclipped_actor_surrogate_cost", "selector_mode": "fixed_nog", "fixed_beta_raw": 0.015, "fixed_gamma_raw": 0.0},
            "unclipped_actor_surrogate_cost",
            "fixed_nog",
        ),
        Candidate(
            "proposed_qp_unclipped_surrogate_actual_selector",
            "proposed_qp_perfLyap",
            1.0,
            1.0,
            1.0,
            {**common, "cost_mode": "unclipped_actor_surrogate_cost", "selector_mode": "actual_surrogate_selector", "fixed_beta_raw": 0.009, "fixed_gamma_raw": 3e-05},
            "unclipped_actor_surrogate_cost",
            "actual_surrogate_selector",
        ),
        Candidate(
            "proposed_qp_unclipped_surrogate_LS_capaware",
            "proposed_qp_perfLyap",
            1.0,
            1.0,
            1.0,
            {**common, "cost_mode": "unclipped_actor_surrogate_cost", "selector_mode": "ls_capaware", "fixed_beta_raw": 0.012, "fixed_gamma_raw": 2.4e-05},
            "unclipped_actor_surrogate_cost",
            "ls_capaware",
        ),
    ]


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    baseline_root = pathlib.Path(args.baseline_root) / "runs_seed0"
    plots_dir = output_root / "plots"
    runs_dir = output_root / "runs_seed0"
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()
    proposed_run_dirs: Dict[str, pathlib.Path] = {}
    for candidate in candidates:
        proposed_run_dirs[candidate.method] = ensure_run(candidate, args, runs_dir)

    baseline_methods = ["adam", "sgd", "egm", "ppm"]
    method_payloads: Dict[str, Dict[str, object]] = {}
    for method in baseline_methods:
        method_payloads[method] = load_baseline_method(baseline_root / method, method)
    for candidate in candidates:
        run_root = runs_dir / candidate.method
        latest_run_dir = proposed_run_dirs[candidate.method]
        method_payloads[candidate.method] = {
            "run_root": run_root,
            "latest_run_dir": latest_run_dir,
            "frames": read_method_frames(candidate.method, latest_run_dir, run_root / "analysis"),
            "summary": pd.read_csv(run_root / "analysis" / "run_summary.csv").iloc[0].to_dict(),
        }

    training_frames = []
    clean_det_frames = []
    control_adv_det_frames = []
    param_frames = []
    update_frames = []
    qp_diag_frames = []
    summary_rows = []
    stochastic_frames = []
    short_audits = []

    for method, payload in method_payloads.items():
        frames = payload["frames"]
        training_frames.append(frames["training"])
        clean_det_frames.append(frames["clean"].assign(eval_type="clean_deterministic"))
        control_adv_det_frames.append(frames["adv"].assign(eval_type="control_adv_deterministic"))
        param_frames.append(frames["param"])
        update_frames.append(pd.concat([frames["protagonist_metrics"], frames["adversary_metrics"]], ignore_index=True))
        if not frames["diagnostics"].empty:
            qp_diag_frames.append(frames["diagnostics"])
        summary = dict(payload["summary"])
        summary["method"] = method
        summary_rows.append(summary)
        stochastic_frames.append(evaluate_stochastic_curves(payload["run_root"], method, device=args.device, n_eval_episodes=args.n_eval_episodes))
        short_audits.append(final_short_return_audits(payload["run_root"], method, env_id=args.env, device=args.device, horizon=args.short_horizon, episodes=args.short_return_episodes, seed=args.seed))

    training_df = pd.concat(training_frames, ignore_index=True)
    eval_df = pd.concat(clean_det_frames + control_adv_det_frames + stochastic_frames, ignore_index=True)
    param_df = pd.concat(param_frames, ignore_index=True)
    update_df = pd.concat(update_frames, ignore_index=True)
    qp_diag_df = pd.concat(qp_diag_frames, ignore_index=True) if qp_diag_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    audit_df = pd.DataFrame(short_audits)
    summary_df = summary_df.merge(audit_df, on="method", how="left")

    if not qp_diag_df.empty:
        agg = (
            qp_diag_df[qp_diag_df["optimizer_role"] == "protagonist"]
            .groupby("method")
            .agg(
                gamma_active_frac=("gamma_active_frac", "mean"),
                fallback_to_noG_frac=("fallback_to_noG", "mean"),
                cap_active_frac=("cap_active", "mean"),
                selected_direction=("direction_mode", "first"),
                selector_mode=("selector_mode", "first"),
                cost_mode=("cost_mode", "first"),
                ls_fit_rank_corr=("ls_fit_rank_corr", "mean"),
                beta_raw_mean=("beta_raw", "mean"),
                gamma_raw_mean=("gamma_raw", "mean"),
                actual_C_change_mean=("actual_C_change", "mean"),
                approx_kl_mean=("approx_kl", "mean"),
                clip_fraction_mean=("clip_fraction", "mean"),
                actor_fraction_of_update_mean=("actor_fraction_of_update", "mean"),
            )
            .reset_index()
        )
        summary_df = summary_df.merge(agg, on="method", how="left")

    training_df.to_csv(output_root / "stage5_training_curves.csv", index=False)
    eval_df.to_csv(output_root / "stage5_eval_curves.csv", index=False)
    param_df.to_csv(output_root / "stage5_param_norms.csv", index=False)
    qp_diag_df.to_csv(output_root / "stage5_qp_diagnostics.csv", index=False)
    summary_df.to_csv(output_root / "stage5_online_summary.csv", index=False)

    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
        "proposed_noG_actor_surrogate": "tab:purple",
        "proposed_qp_actor_surrogate_actual_selector": "tab:blue",
        "proposed_qp_actor_surrogate_LS_capaware": "tab:cyan",
        "proposed_noG_unclipped_surrogate": "tab:brown",
        "proposed_qp_unclipped_surrogate_actual_selector": "tab:pink",
        "proposed_qp_unclipped_surrogate_LS_capaware": "tab:olive",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_training(ax, training_df, colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_training_return.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "clean_deterministic", "Clean deterministic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_clean_det_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "clean_stochastic", "Clean stochastic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_clean_stoch_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "control_adv_deterministic", "Control-adv deterministic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_control_adv_det_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "control_adv_stochastic", "Control-adv stochastic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_control_adv_stoch_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not qp_diag_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_qp_metric(ax, qp_diag_df, "gamma_raw", "Gamma raw", colors)
        plot_qp_metric(ax, qp_diag_df, "beta_raw", "Beta raw", colors)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plots_dir / "stage5_beta_gamma.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_qp_metric(ax, qp_diag_df, "fallback_to_noG", "Fallback-to-noG rate", colors)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plots_dir / "stage5_fallback_rate.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        (plots_dir / "stage5_beta_gamma.png").write_text("", encoding="utf-8")
        (plots_dir / "stage5_fallback_rate.png").write_text("", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_update_norms(ax, update_df, colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_update_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_param_norms(ax, param_df, colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_param_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    final_rows = []
    for method in summary_df["method"]:
        clean_det = eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "clean_deterministic")].sort_values("outer_iteration")
        clean_stoch = eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "clean_stochastic")].sort_values("outer_iteration")
        adv_det = eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "control_adv_deterministic")].sort_values("outer_iteration")
        adv_stoch = eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "control_adv_stochastic")].sort_values("outer_iteration")
        final_rows.append(
            {
                "method": method,
                "clean_det_final": float(clean_det["mean_reward"].iloc[-1]) if not clean_det.empty else np.nan,
                "clean_stoch_final": float(clean_stoch["mean_reward"].iloc[-1]) if not clean_stoch.empty else np.nan,
                "control_adv_det_final": float(adv_det["mean_reward"].iloc[-1]) if not adv_det.empty else np.nan,
                "control_adv_stoch_final": float(adv_stoch["mean_reward"].iloc[-1]) if not adv_stoch.empty else np.nan,
            }
        )
    final_bar_df = pd.DataFrame(final_rows)
    summary_df = summary_df.merge(final_bar_df, on="method", how="left")
    summary_df.to_csv(output_root / "stage5_online_summary.csv", index=False)

    order = list(summary_df["method"])
    x = np.arange(len(order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 6))
    final_map = final_bar_df.set_index("method")
    ax.bar(x - 1.5 * width, [final_map.loc[m, "clean_det_final"] for m in order], width=width, label="clean_det")
    ax.bar(x - 0.5 * width, [final_map.loc[m, "clean_stoch_final"] for m in order], width=width, label="clean_stoch")
    ax.bar(x + 0.5 * width, [final_map.loc[m, "control_adv_det_final"] for m in order], width=width, label="control_adv_det")
    ax.bar(x + 1.5 * width, [final_map.loc[m, "control_adv_stoch_final"] for m in order], width=width, label="control_adv_stoch")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Mean return")
    ax.set_title("Final comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_final_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    collage_paths = [
        plots_dir / "stage5_training_return.png",
        plots_dir / "stage5_clean_det_eval.png",
        plots_dir / "stage5_clean_stoch_eval.png",
        plots_dir / "stage5_control_adv_det_eval.png",
        plots_dir / "stage5_control_adv_stoch_eval.png",
        plots_dir / "stage5_beta_gamma.png",
        plots_dir / "stage5_fallback_rate.png",
        plots_dir / "stage5_update_norms.png",
        plots_dir / "stage5_param_norms.png",
        plots_dir / "stage5_final_bar.png",
    ]
    make_collage(collage_paths, plots_dir / "stage5_all_plots_big.png", cols=2)

    def final_metric(method: str, col: str) -> float:
        sub = summary_df[summary_df["method"] == method]
        return float(sub[col].iloc[0]) if not sub.empty and col in sub.columns else float("nan")

    actor_actual = "proposed_qp_actor_surrogate_actual_selector"
    actor_ls = "proposed_qp_actor_surrogate_LS_capaware"
    unclipped_actual = "proposed_qp_unclipped_surrogate_actual_selector"
    unclipped_ls = "proposed_qp_unclipped_surrogate_LS_capaware"
    noG_actor = "proposed_noG_actor_surrogate"
    noG_unclipped = "proposed_noG_unclipped_surrogate"

    actor_best = actor_actual if final_metric(actor_actual, "clean_det_final") >= final_metric(actor_ls, "clean_det_final") else actor_ls
    unclipped_best = unclipped_actual if final_metric(unclipped_actual, "clean_det_final") >= final_metric(unclipped_ls, "clean_det_final") else unclipped_ls
    selector_winner = "actor_surrogate" if final_metric(actor_best, "clean_det_final") >= final_metric(unclipped_best, "clean_det_final") else "unclipped_actor_surrogate"

    lines = [
        "# Stage 5 Online Screen Report",
        "",
        f"- Proposed methods run: `{len(candidates)}`",
        f"- Baselines read from: `{baseline_root}`",
        f"- Training budget: `total_iterations={args.iterations}`, `seed={args.seed}`",
        f"- Eval schedule target: `eval_freq={args.eval_freq}`, `n_eval_episodes={args.n_eval_episodes}`",
        "",
        "## Required Answers",
        "",
        f"1. Which online selector works better: actor_surrogate or unclipped_actor_surrogate? `{selector_winner}` on final clean deterministic eval.",
        f"2. Does QP beat noG under the same selector? actor_surrogate: `{'Yes' if max(final_metric(actor_actual, 'clean_det_final'), final_metric(actor_ls, 'clean_det_final')) > final_metric(noG_actor, 'clean_det_final') else 'No'}`; unclipped: `{'Yes' if max(final_metric(unclipped_actual, 'clean_det_final'), final_metric(unclipped_ls, 'clean_det_final')) > final_metric(noG_unclipped, 'clean_det_final') else 'No'}`.",
        f"3. Does either proposed method beat SGD/EGM/PPM? best proposed clean_det=`{max(summary_df['clean_det_final']):.6g}`, SGD=`{final_metric('sgd', 'clean_det_final'):.6g}`, EGM=`{final_metric('egm', 'clean_det_final'):.6g}`, PPM=`{final_metric('ppm', 'clean_det_final'):.6g}`.",
        f"4. Do short_clean_return and short_rarl_return agree with the training surrogate? Compare `short_clean_return_cost`, `short_rarl_return_cost`, and final clean/control curves in `stage5_online_summary.csv`.",
        "",
        "## Notes",
        "",
        "- short_clean_return_cost and short_rarl_return_cost are final-checkpoint audits only in this stage.",
        "- Stochastic clean/control curves were computed post-hoc from saved checkpoints on the same eval schedule.",
        "",
        summary_df.to_csv(index=False),
    ]
    (output_root / "stage5_online_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
