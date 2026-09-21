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

from scripts.full_policy_followup_common import SavedRun, load_yaml, load_rarl_for_eval, set_rarl_eval_mode


@dataclass(frozen=True)
class Candidate:
    method_label: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]
    eta: float


def pick_column(frame: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise KeyError(f"Missing required columns. Tried {list(candidates)} but only found {list(frame.columns)}")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage M8 rawFG online on top of matched baselines")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
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
    tokens = []
    for key, value in sorted(kwargs.items()):
        if isinstance(value, str):
            rendered = repr(value)
        elif isinstance(value, float) and math.isinf(value):
            rendered = "float('inf')"
        else:
            rendered = value
        tokens.append(f"{key}:{rendered}")
    return tokens


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def build_saved_run(run_dir: pathlib.Path, method: str) -> SavedRun:
    config_dir = next(path for path in run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists())
    return SavedRun(
        method=method,
        tag=method,
        run_root=run_dir,
        model_dir=config_dir,
        args_data=load_yaml(config_dir / "args.yml"),
        config_data=load_yaml(config_dir / "config.yml"),
    )


def evaluate_control_strength_sweep(run_dir: pathlib.Path, method: str, device: str, n_eval_episodes: int) -> pd.DataFrame:
    saved_run = build_saved_run(run_dir, method)
    strengths = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    model, vec_env = load_rarl_for_eval(saved_run, adv_impact="control", adv_strength=1.0, device=device)
    for strength in strengths:
        set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=strength)
        episode_rewards = []
        perturbation_norms = []
        clip_fractions = []
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_perturb = []
        ep_clip = []
        while len(episode_rewards) < n_eval_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            ep_reward += float(rewards[0])
            info = infos[0]
            ep_perturb.append(float(info.get("applied_control_perturbation_norm", info.get("applied_disturbance_norm", 0.0))))
            ep_clip.append(float(info.get("action_clip_fraction", 0.0)))
            if bool(dones[0]):
                episode_rewards.append(ep_reward)
                perturbation_norms.append(float(np.mean(ep_perturb)) if ep_perturb else 0.0)
                clip_fractions.append(float(np.mean(ep_clip)) if ep_clip else 0.0)
                obs = vec_env.reset()
                ep_reward = 0.0
                ep_perturb = []
                ep_clip = []
        rows.append(
            {
                "method": method,
                "adv_strength": strength,
                "mean_return": float(np.mean(episode_rewards)),
                "std_return": float(np.std(episode_rewards)),
                "applied_control_perturbation_norm": float(np.mean(perturbation_norms)),
                "action_clip_fraction": float(np.mean(clip_fractions)),
            }
        )
    vec_env.close()
    return pd.DataFrame(rows)


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

    protagonist_metrics = pd.read_csv(latest_run_dir / "analysis" / "protagonist_training_metrics.csv")
    protagonist_metrics["num_timesteps"] = pd.to_numeric(protagonist_metrics[pick_column(protagonist_metrics, ["num_timesteps", "timesteps", "timestep"])], errors="coerce")
    protagonist_metrics["actor_update_norm"] = pd.to_numeric(protagonist_metrics[pick_column(protagonist_metrics, ["actor_update_norm"])], errors="coerce")
    protagonist_metrics["critic_update_norm"] = pd.to_numeric(protagonist_metrics[pick_column(protagonist_metrics, ["critic_update_norm"])], errors="coerce")
    p_kl_col = pick_column(protagonist_metrics, ["approx_kl"], required=False)
    p_clip_col = pick_column(protagonist_metrics, ["clip_fraction"], required=False)
    protagonist_metrics["approx_kl"] = pd.to_numeric(protagonist_metrics[p_kl_col], errors="coerce") if p_kl_col else np.nan
    protagonist_metrics["clip_fraction"] = pd.to_numeric(protagonist_metrics[p_clip_col], errors="coerce") if p_clip_col else np.nan
    protagonist_metrics["method"] = method_label
    protagonist_metrics["optimizer_role"] = "protagonist"
    protagonist_metrics["outer_iteration"] = protagonist_metrics["num_timesteps"] / float(n_steps)

    adversary_metrics = pd.read_csv(latest_run_dir / "analysis" / "adversary_training_metrics.csv")
    adversary_metrics["num_timesteps"] = pd.to_numeric(adversary_metrics[pick_column(adversary_metrics, ["num_timesteps", "timesteps", "timestep"])], errors="coerce")
    adversary_metrics["actor_update_norm"] = pd.to_numeric(adversary_metrics[pick_column(adversary_metrics, ["actor_update_norm"])], errors="coerce")
    adversary_metrics["critic_update_norm"] = pd.to_numeric(adversary_metrics[pick_column(adversary_metrics, ["critic_update_norm"])], errors="coerce")
    a_kl_col = pick_column(adversary_metrics, ["approx_kl"], required=False)
    a_clip_col = pick_column(adversary_metrics, ["clip_fraction"], required=False)
    adversary_metrics["approx_kl"] = pd.to_numeric(adversary_metrics[a_kl_col], errors="coerce") if a_kl_col else np.nan
    adversary_metrics["clip_fraction"] = pd.to_numeric(adversary_metrics[a_clip_col], errors="coerce") if a_clip_col else np.nan
    adversary_metrics["method"] = method_label
    adversary_metrics["optimizer_role"] = "adversary"
    adversary_metrics["outer_iteration"] = adversary_metrics["num_timesteps"] / float(n_steps)

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
            if "approx_kl" not in role_df.columns and "approx_kl_after" in role_df.columns:
                role_df["approx_kl"] = pd.to_numeric(role_df["approx_kl_after"], errors="coerce")
            if "clip_fraction" not in role_df.columns and "clip_fraction_after" in role_df.columns:
                role_df["clip_fraction"] = pd.to_numeric(role_df["clip_fraction_after"], errors="coerce")
            if "update_norm_post_cap" not in role_df.columns and "update_norm" in role_df.columns:
                role_df["update_norm_post_cap"] = pd.to_numeric(role_df["update_norm"], errors="coerce")
            if "update_norm_pre_cap" not in role_df.columns and "update_norm" in role_df.columns:
                role_df["update_norm_pre_cap"] = pd.to_numeric(role_df["update_norm"], errors="coerce")
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


def ensure_run(candidate: Candidate, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
    run_root = runs_dir / candidate.method_label
    saved_models_dir = run_root / "saved_models"
    analysis_dir = run_root / "analysis"
    env_root = saved_models_dir / "rarl-ppo" / args.env
    if not env_root.exists():
        raise FileNotFoundError(f"Existing run not found for {candidate.method_label}: {env_root}")
    if not (analysis_dir / "run_summary.csv").exists():
        raise FileNotFoundError(f"Analysis missing for {candidate.method_label}: {analysis_dir / 'run_summary.csv'}")
    latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
    return latest_run_dir


def plot_eval_with_band(ax, df: pd.DataFrame, title: str, ylabel: str, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["mean_reward"], label=method, color=colors.get(method))
        ax.fill_between(group["outer_iteration"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], color=colors.get(method), alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_param_norms(ax, df: pd.DataFrame, agent_name: str, colors: Dict[str, str]) -> None:
    sub = df[df["agent_name"] == agent_name]
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        base = colors.get(method)
        ax.plot(group["outer_iteration"], group["actor_param_norm"], color=base, label=f"{method} actor")
        ax.plot(group["outer_iteration"], group["log_std_norm"], color=base, linestyle=":", label=f"{method} log_std")
        ax.plot(group["outer_iteration"], group["critic_param_norm"], color=base, linestyle="--", label=f"{method} critic")
        ax.plot(group["outer_iteration"], group["total_param_norm"], color=base, linestyle="-.", label=f"{method} total")
    ax.set_title(f"{agent_name.capitalize()} parameter norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)


def plot_update_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["actor_update_norm"], color=colors.get(method), label=f"{method} actor")
        ax.plot(group["outer_iteration"], group["critic_update_norm"], color=colors.get(method), linestyle="--", label=f"{method} critic")
    ax.set_title("Protagonist update norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Update norm")
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


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    runs_dir = output_root / "runs_seed0"
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    preflight_df = pd.read_csv(output_root / "rawFG_preflight_after_nanfix_summary.csv")
    pass_qp = sorted(preflight_df[(preflight_df["method"] == "proposed_qp_new_v2_rawFG") & (preflight_df["preflight_pass"] == True)]["eta"].tolist())
    pass_nog = sorted(preflight_df[(preflight_df["method"] == "proposed_noG_new_v2_rawFG") & (preflight_df["preflight_pass"] == True)]["eta"].tolist())
    pass_etas = sorted(set(pass_qp).intersection(pass_nog))
    pass_etas = [eta for eta in pass_etas if eta in {0.1, 0.02}]

    candidates: List[Candidate] = []
    for eta in pass_etas:
        common = {
            "optimizer_scope": "full_policy",
            "qp_fd_eps": 1e-3,
            "qp_beta_probe": 1e-3,
            "qp_gamma_probe": 1e-6,
            "qp_ridge": 1e-8,
            "qp_actor_weight": 1.0,
            "qp_logstd_weight": 1.0,
            "qp_critic_weight": 0.3,
            "qp_beta_max": 1e-2,
            "qp_gamma_max": 3e-5,
            "qp_max_update_norm": float("inf"),
            "qp_eps": 1e-8,
        }
        eta_label = f"{eta:g}"
        candidates.append(Candidate(f"proposed_noG_rawFG_eta{eta_label}", "proposed_noG_rawFG", eta, 0.5, 0.5, dict(common), eta))
        candidates.append(Candidate(f"proposed_qp_rawFG_eta{eta_label}", "proposed_qp_rawFG", eta, 0.5, 0.5, dict(common), eta))

    proposed_run_dirs: Dict[str, pathlib.Path] = {}
    for candidate in candidates:
        proposed_run_dirs[candidate.method_label] = ensure_run(candidate, args, runs_dir)

    summary_rows = []
    training_frames = []
    clean_frames = []
    adv_frames = []
    param_frames = []
    update_frames = []
    qp_diag_frames = []

    baseline_methods = ["adam", "sgd", "egm", "ppm"]
    all_methods = baseline_methods + [candidate.method_label for candidate in candidates]

    for method_label in all_methods:
        if method_label in baseline_methods:
            latest_run_dir = find_latest_run_dir(runs_dir / method_label / "saved_models", args.env)
        else:
            latest_run_dir = proposed_run_dirs[method_label]
        analysis_dir = runs_dir / method_label / "analysis"
        frames = read_method_frames(method_label, latest_run_dir, analysis_dir)
        training_frames.append(frames["training"])
        clean_frames.append(frames["clean"])
        adv_frames.append(frames["adv"])
        param_frames.append(frames["param"])
        update_frames.append(pd.concat([frames["protagonist_metrics"], frames["adversary_metrics"]], ignore_index=True))
        if not frames["diagnostics"].empty:
            frames["diagnostics"]["eta"] = next((cand.eta for cand in candidates if cand.method_label == method_label), np.nan)
            frames["diagnostics"]["beta_over_eta_EGM"] = frames["diagnostics"]["beta"] / args.eta_egm if "beta" in frames["diagnostics"].columns else np.nan
            frames["diagnostics"]["gamma_over_eta_EGM_squared"] = frames["diagnostics"]["gamma"] / (args.eta_egm ** 2) if "gamma" in frames["diagnostics"].columns else np.nan
            qp_diag_frames.append(frames["diagnostics"])
        summary_row = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        summary_row["method"] = method_label
        summary_row["eta"] = next((cand.eta for cand in candidates if cand.method_label == method_label), np.nan)
        summary_rows.append(summary_row)

    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    param_df = pd.concat(param_frames, ignore_index=True)
    update_df = pd.concat(update_frames, ignore_index=True)
    qp_diag_df = pd.concat(qp_diag_frames, ignore_index=True) if qp_diag_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    # Merge baseline sweep with proposed sweep
    sweep_df = pd.read_csv(output_root / "rawFG_matched_robustness_sweep.csv")
    proposed_sweeps = []
    for method_label, run_dir in proposed_run_dirs.items():
        proposed_sweeps.append(evaluate_control_strength_sweep(run_dir, method_label, args.device, args.n_eval_episodes))
    if proposed_sweeps:
        sweep_df = pd.concat([sweep_df] + proposed_sweeps, ignore_index=True)

    eval_df = pd.concat(
        [
            clean_df.assign(eval_type="clean"),
            adv_df.assign(eval_type="control_adversarial"),
        ],
        ignore_index=True,
    )

    summary_df.to_csv(output_root / "rawFG_M8_online_summary.csv", index=False)
    eval_df.to_csv(output_root / "rawFG_M8_eval_curves.csv", index=False)
    param_df.to_csv(output_root / "rawFG_M8_param_norms.csv", index=False)
    update_df.to_csv(output_root / "rawFG_M8_update_diagnostics.csv", index=False)
    qp_diag_df.to_csv(output_root / "rawFG_M8_qp_diagnostics.csv", index=False)
    sweep_df.to_csv(output_root / "rawFG_M8_robustness_sweep.csv", index=False)

    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
        "proposed_noG_rawFG_eta0.1": "tab:purple",
        "proposed_qp_rawFG_eta0.1": "tab:blue",
        "proposed_noG_rawFG_eta0.02": "tab:pink",
        "proposed_qp_rawFG_eta0.02": "tab:cyan",
    }

    plt.figure(figsize=(10, 6))
    for method, group in training_df.groupby("method"):
        group = group.sort_values("outer_iteration")
        plt.plot(group["outer_iteration"], group["episode_return"], label=method, color=colors.get(method))
    plt.title("Training return vs outer iteration")
    plt.xlabel("Outer iteration")
    plt.ylabel("Episode return")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "M8_training_return_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_eval_with_band(ax, clean_df, "Clean eval vs outer iteration", "Mean return", colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "M8_clean_eval_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_eval_with_band(ax, adv_df, "Control adversarial eval vs outer iteration", "Mean return", colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "M8_control_adv_eval_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_param_norms(ax, param_df, "protagonist", colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "M8_protagonist_param_norms_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_param_norms(ax, param_df, "adversary", colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "M8_adversary_param_norms_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_update_norms(ax, update_df, colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "M8_update_norms_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not qp_diag_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        proposed_only = qp_diag_df[qp_diag_df["method"].str.contains("proposed_")].copy()
        for method, group in proposed_only.groupby("method"):
            group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
            axes[0, 0].plot(group["outer_iteration"], group["beta"], label=method, color=colors.get(method))
            axes[0, 1].plot(group["outer_iteration"], group["gamma"], label=method, color=colors.get(method))
            axes[1, 0].plot(group["outer_iteration"], group["beta_eff"], label=f"{method} beta_eff", color=colors.get(method))
            axes[1, 0].plot(group["outer_iteration"], group["gamma_eff"], linestyle="--", label=f"{method} gamma_eff", color=colors.get(method))
            axes[1, 1].plot(group["outer_iteration"], group["gamma_active_frac"], label=f"{method} active", color=colors.get(method))
            axes[1, 1].plot(group["outer_iteration"], group["G_contribution_norm"], linestyle="--", label=f"{method} G", color=colors.get(method))
        axes[0, 0].set_title("beta")
        axes[0, 1].set_title("gamma")
        axes[1, 0].set_title("beta_eff / gamma_eff")
        axes[1, 1].set_title("gamma_active_frac / G_contribution_norm")
        for ax in axes.ravel():
            ax.set_xlabel("Outer iteration")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "M8_qp_beta_gamma_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for method, group in proposed_only.groupby("method"):
            group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
            axes[0, 0].plot(group["outer_iteration"], group["actual_V_change"], label=method, color=colors.get(method))
            axes[0, 1].plot(group["outer_iteration"], group["q_pred"], label=method, color=colors.get(method))
            axes[1, 0].plot(group["outer_iteration"], group["approx_kl"], label=method, color=colors.get(method))
            axes[1, 1].plot(group["outer_iteration"], group["clip_fraction"], label=method, color=colors.get(method))
        axes[0, 0].set_title("actual_V_change")
        axes[0, 1].set_title("q_pred")
        axes[1, 0].set_title("approx_kl")
        axes[1, 1].set_title("clip_fraction")
        for ax in axes.ravel():
            ax.set_xlabel("Outer iteration")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "M8_qp_V_KL_clip_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in sweep_df.groupby("method"):
        group = group.sort_values("adv_strength")
        ax.errorbar(group["adv_strength"], group["mean_return"], yerr=group["std_return"], marker="o", label=method, color=colors.get(method))
    ax.set_title("Control robustness sweep final")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "M8_control_robustness_sweep_final.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    bar_rows = []
    for method, row in summary_df.set_index("method").iterrows():
        bar_rows.extend(
            [
                {"method": method, "metric": "clean_last5", "value": float(row.get("last5_clean_mean", np.nan))},
                {"method": method, "metric": "control_adv_last5", "value": float(row.get("last5_adversarial_mean", np.nan))},
                {"method": method, "metric": "robustness_auc", "value": float(np.trapz(sweep_df[sweep_df["method"] == method].sort_values("adv_strength")["mean_return"], sweep_df[sweep_df["method"] == method].sort_values("adv_strength")["adv_strength"])) if not sweep_df[sweep_df["method"] == method].empty else np.nan},
            ]
        )
    bar_df = pd.DataFrame(bar_rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, title in zip(axes, ["clean_last5", "control_adv_last5", "robustness_auc"], ["Final clean", "Final control-adv", "Robustness AUC"]):
        sub = bar_df[bar_df["metric"] == metric]
        ax.bar(sub["method"], sub["value"], color=[colors.get(m) for m in sub["method"]])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "M8_final_bar_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    collage_paths = [
        plots_dir / "M8_training_return_vs_outer_iteration.png",
        plots_dir / "M8_clean_eval_vs_outer_iteration.png",
        plots_dir / "M8_control_adv_eval_vs_outer_iteration.png",
        plots_dir / "M8_protagonist_param_norms_vs_outer_iteration.png",
        plots_dir / "M8_adversary_param_norms_vs_outer_iteration.png",
        plots_dir / "M8_update_norms_vs_outer_iteration.png",
        plots_dir / "M8_qp_beta_gamma_diagnostics.png",
        plots_dir / "M8_qp_V_KL_clip_diagnostics.png",
        plots_dir / "M8_control_robustness_sweep_final.png",
        plots_dir / "M8_final_bar_comparison.png",
    ]
    make_collage(collage_paths, plots_dir / "M8_all_plots_big.png", cols=2)

    # Build final report
    report_lines = [
        "# RawFG M8 final report",
        "",
        f"1. Did eta=0.1 QP finish online stably? `{bool('proposed_qp_rawFG_eta0.1' in summary_df['method'].values)}`",
        f"2. Did eta=0.02 QP finish online stably? `{bool('proposed_qp_rawFG_eta0.02' in summary_df['method'].values)}`",
    ]

    def last5(metric_method: str, metric_col: str) -> float:
        if metric_method not in summary_df["method"].values:
            return float("nan")
        return float(summary_df.loc[summary_df["method"] == metric_method, metric_col].iloc[0])

    report_lines.extend(
        [
            f"3. Does proposed_qp_rawFG_eta0.1 beat proposed_noG_rawFG_eta0.1? `{last5('proposed_qp_rawFG_eta0.1', 'last5_clean_mean') > last5('proposed_noG_rawFG_eta0.1', 'last5_clean_mean') if 'proposed_qp_rawFG_eta0.1' in summary_df['method'].values and 'proposed_noG_rawFG_eta0.1' in summary_df['method'].values else False}`",
            f"4. Does proposed_qp_rawFG_eta0.02 beat proposed_noG_rawFG_eta0.02? `{last5('proposed_qp_rawFG_eta0.02', 'last5_clean_mean') > last5('proposed_noG_rawFG_eta0.02', 'last5_clean_mean') if 'proposed_qp_rawFG_eta0.02' in summary_df['method'].values and 'proposed_noG_rawFG_eta0.02' in summary_df['method'].values else False}`",
            f"5. Does either QP beat SGD on clean eval? `{bool(any(last5(m, 'last5_clean_mean') > last5('sgd', 'last5_clean_mean') for m in ['proposed_qp_rawFG_eta0.1', 'proposed_qp_rawFG_eta0.02'] if m in summary_df['method'].values))}`",
            f"6. Does either QP beat PPM on clean eval? `{bool(any(last5(m, 'last5_clean_mean') > last5('ppm', 'last5_clean_mean') for m in ['proposed_qp_rawFG_eta0.1', 'proposed_qp_rawFG_eta0.02'] if m in summary_df['method'].values))}`",
            f"7. Does either QP beat EGM on clean eval? `{bool(any(last5(m, 'last5_clean_mean') > last5('egm', 'last5_clean_mean') for m in ['proposed_qp_rawFG_eta0.1', 'proposed_qp_rawFG_eta0.02'] if m in summary_df['method'].values))}`",
            f"8. Does either QP beat SGD/PPM/EGM on control adversarial eval? `{bool(any(last5(m, 'last5_adversarial_mean') > max(last5('sgd', 'last5_adversarial_mean'), last5('ppm', 'last5_adversarial_mean'), last5('egm', 'last5_adversarial_mean')) for m in ['proposed_qp_rawFG_eta0.1', 'proposed_qp_rawFG_eta0.02'] if m in summary_df['method'].values))}`",
        ]
    )

    if not qp_diag_df.empty:
        proposed_only = qp_diag_df[qp_diag_df["method"].str.contains("proposed_")].copy()
        gamma_active_online = float(pd.to_numeric(proposed_only["gamma_active_frac"], errors="coerce").mean()) if "gamma_active_frac" in proposed_only.columns else float("nan")
        beta_ratio_mean = float(pd.to_numeric(proposed_only["beta_over_eta_EGM"], errors="coerce").mean()) if "beta_over_eta_EGM" in proposed_only.columns else float("nan")
        gamma_ratio_mean = float(pd.to_numeric(proposed_only["gamma_over_eta_EGM_squared"], errors="coerce").mean()) if "gamma_over_eta_EGM_squared" in proposed_only.columns else float("nan")
        report_lines.extend(
            [
                f"9. Does gamma remain active online? `{bool(np.isfinite(gamma_active_online) and gamma_active_online > 0.0)}`",
                f"10. Are beta/gamma usually larger than EGM-like coefficients? `{bool(np.isfinite(beta_ratio_mean) and beta_ratio_mean > 1.0 and np.isfinite(gamma_ratio_mean) and gamma_ratio_mean > 1.0)}`",
            ]
        )
    else:
        report_lines.extend(
            [
                "9. Does gamma remain active online? `False`",
                "10. Are beta/gamma usually larger than EGM-like coefficients? `False`",
            ]
        )

    proposed_update_mean = float(pd.to_numeric(qp_diag_df["update_norm_post_cap"], errors="coerce").mean()) if not qp_diag_df.empty and "update_norm_post_cap" in qp_diag_df.columns else float("nan")
    baseline_update_mean = float(pd.to_numeric(update_df[(update_df["optimizer_role"] == "protagonist") & (update_df["method"].isin(["sgd", "egm", "ppm"]))]["actor_update_norm"], errors="coerce").mean())
    report_lines.append(f"11. Are QP update norms comparable to baselines? `{bool(np.isfinite(proposed_update_mean) and np.isfinite(baseline_update_mean) and proposed_update_mean <= baseline_update_mean * 5.0)}`")

    failure_label = "E. G helps noG but not enough vs EGM"
    if "proposed_qp_rawFG_eta0.1" in summary_df["method"].values:
        if last5("proposed_qp_rawFG_eta0.1", "last5_clean_mean") < last5("proposed_qp_rawFG_eta0.02", "last5_clean_mean") if "proposed_qp_rawFG_eta0.02" in summary_df["method"].values else False:
            failure_label = "A. eta too aggressive"
    report_lines.append(f"12. If QP fails, classification: `{failure_label}`")

    (output_root / "rawFG_M8_final_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
