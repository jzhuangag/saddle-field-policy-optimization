from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import SavedRun, load_yaml, load_rarl_for_eval, set_rarl_eval_mode


@dataclass(frozen=True)
class Candidate:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 8 scope-by-method benchmark")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--stage8c-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--scope", type=str, required=True, choices=["full_policy", "actor_game"])
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def load_stage5_configs(stage5_root: pathlib.Path) -> Dict[str, Dict[str, object]]:
    return json.loads((stage5_root / "final_full_policy_optimizer_configs.json").read_text(encoding="utf-8"))


def choose_ppm_config(configs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    ppm_cfg = configs["ppm"]
    finalists = list(ppm_cfg.get("finalists", []))
    for candidate in finalists:
        inner_steps = int(candidate.get("optimizer_kwargs", {}).get("inner_steps", 1))
        if inner_steps > 2:
            return candidate
    return ppm_cfg


def load_candidates(stage5_root: pathlib.Path, stage8c_root: pathlib.Path, scope: str) -> List[Candidate]:
    configs = load_stage5_configs(stage5_root)
    ppm_cfg = choose_ppm_config(configs)
    egm_cfg = configs["egm"]
    sgd_cfg = configs["sgd"]
    adam_cfg = configs["adam"]
    selected = json.loads((stage8c_root / "stage8_selected_qp_settings.json").read_text(encoding="utf-8"))[scope]
    qp_kwargs = {
        "optimizer_scope": scope,
        "qp_normalization": selected["qp_normalization"],
        "qp_g_alpha": float(selected["qp_g_alpha"]),
        "qp_eps": 1e-8,
    }
    return [
        Candidate("adam", "adam", float(adam_cfg["lr"]), float(adam_cfg["max_grad_norm"]), float(adam_cfg["vf_coef"]), dict(adam_cfg.get("optimizer_kwargs", {}))),
        Candidate("sgd", "sgd", float(sgd_cfg["lr"]), float(sgd_cfg["max_grad_norm"]), float(sgd_cfg["vf_coef"]), dict(sgd_cfg.get("optimizer_kwargs", {}))),
        Candidate("egm", "egm", float(egm_cfg["lr"]), float(egm_cfg["max_grad_norm"]), float(egm_cfg["vf_coef"]), dict(egm_cfg.get("optimizer_kwargs", {}))),
        Candidate("ppm", "ppm", float(ppm_cfg["lr"]), float(ppm_cfg["max_grad_norm"]), float(ppm_cfg["vf_coef"]), dict(ppm_cfg.get("optimizer_kwargs", {}))),
        Candidate("proposed_noG", "proposed_noG", float(egm_cfg["lr"]), float(egm_cfg["max_grad_norm"]), float(egm_cfg["vf_coef"]), dict(qp_kwargs)),
        Candidate("proposed_qp", "proposed_qp", float(egm_cfg["lr"]), float(egm_cfg["max_grad_norm"]), float(egm_cfg["vf_coef"]), dict(qp_kwargs)),
    ]


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def render_kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens = []
    for key, value in sorted(kwargs.items()):
        if isinstance(value, float) and not np.isfinite(value):
            continue
        rendered = repr(value) if isinstance(value, str) else value
        tokens.append(f"{key}:{rendered}")
    return tokens


def ensure_run(candidate: Candidate, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
    run_root = runs_dir / candidate.method
    saved_models_dir = run_root / "saved_models"
    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    analysis_dir = run_root / "analysis"
    run_root.mkdir(parents=True, exist_ok=True)

    latest_run_dir = None
    env_root = saved_models_dir / "rarl-ppo" / args.env
    if env_root.exists():
        latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
    control_proxy_npz = None if latest_run_dir is None else latest_run_dir / "control_proxy_eval" / "evaluations.npz"
    if latest_run_dir is not None and control_proxy_npz is not None and control_proxy_npz.exists():
        stdout_candidate = run_root / "stdout.txt"
        stderr_candidate = run_root / "stderr.txt"
        if (analysis_dir / "run_summary.csv").exists():
            return latest_run_dir
        analysis_dir.mkdir(parents=True, exist_ok=True)
        if stdout_candidate.exists():
            shutil.copy2(stdout_candidate, analysis_dir / "stdout.txt")
        else:
            (analysis_dir / "stdout.txt").write_text("", encoding="utf-8")
        if stderr_candidate.exists():
            shutil.copy2(stderr_candidate, analysis_dir / "stderr.txt")
        else:
            (analysis_dir / "stderr.txt").write_text("", encoding="utf-8")
        run_command(
            [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.method],
            cwd=pathlib.Path(args.repo_dir),
            stdout_path=analysis_dir / "analyze_stdout.txt",
            stderr_path=analysis_dir / "analyze_stderr.txt",
        )
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
        "force",
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
        "--control-proxy-eval",
        "--saved-models-path",
        str(saved_models_dir),
        "--log-folder",
        str(logging_dir),
        "--tensorboard-log",
        str(tb_dir),
        "--optimizer-scope",
        args.scope,
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
    protagonist_tokens = render_kwargs_tokens(candidate.optimizer_kwargs)
    adversary_tokens = render_kwargs_tokens(candidate.optimizer_kwargs)
    if protagonist_tokens:
        command.extend(["--protagonist-optimizer-kwargs", *protagonist_tokens])
    if adversary_tokens:
        command.extend(["--adversary-optimizer-kwargs", *adversary_tokens])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
    latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.method],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return latest_run_dir


def evaluate_strength_sweep(run_dir: pathlib.Path, method: str, scope: str, device: str, n_eval_episodes: int) -> pd.DataFrame:
    config_dir = next(path for path in run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists())
    saved_run = SavedRun(
        method=method,
        tag=method,
        run_root=run_dir,
        model_dir=config_dir,
        args_data=load_yaml(config_dir / "args.yml"),
        config_data=load_yaml(config_dir / "config.yml"),
    )
    strengths = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for adv_impact in ["force", "control"]:
        model, vec_env = load_rarl_for_eval(saved_run, adv_impact=adv_impact, adv_strength=1.0, device=device)
        for strength in strengths:
            set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=strength)
            episode_rewards = []
            episode_force = []
            episode_disturbance = []
            episode_adv_pre = []
            episode_adv_post = []
            episode_state = []
            obs = vec_env.reset()
            ep_reward = 0.0
            force_values: List[float] = []
            disturbance_values: List[float] = []
            adv_pre_values: List[float] = []
            adv_post_values: List[float] = []
            state_values: List[float] = []
            while len(episode_rewards) < n_eval_episodes:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = vec_env.step(action)
                ep_reward += float(rewards[0])
                info = infos[0]
                force_values.append(float(info.get("applied_force_norm", 0.0)))
                disturbance_values.append(float(info.get("applied_disturbance_norm", 0.0)))
                adv_pre_values.append(float(info.get("adversary_action_norm_pre_clip", 0.0)))
                adv_post_values.append(float(info.get("adversary_action_norm_post_clip", 0.0)))
                state_values.append(float(info.get("state_norm", 0.0)))
                if bool(dones[0]):
                    episode_rewards.append(ep_reward)
                    episode_force.append(float(np.mean(force_values)) if force_values else 0.0)
                    episode_disturbance.append(float(np.mean(disturbance_values)) if disturbance_values else 0.0)
                    episode_adv_pre.append(float(np.mean(adv_pre_values)) if adv_pre_values else 0.0)
                    episode_adv_post.append(float(np.mean(adv_post_values)) if adv_post_values else 0.0)
                    episode_state.append(float(np.mean(state_values)) if state_values else 0.0)
                    ep_reward = 0.0
                    force_values = []
                    disturbance_values = []
                    adv_pre_values = []
                    adv_post_values = []
                    state_values = []
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "adv_impact": "control-proxy" if adv_impact == "control" else "force",
                    "adv_strength": strength,
                    "mean_return": float(np.mean(episode_rewards)),
                    "std_return": float(np.std(episode_rewards)),
                    "applied_force_norm": float(np.mean(episode_force)),
                    "applied_disturbance_norm": float(np.mean(episode_disturbance)),
                    "adversary_action_norm_pre_clip": float(np.mean(episode_adv_pre)),
                    "adversary_action_norm_post_clip": float(np.mean(episode_adv_post)),
                    "state_norm": float(np.mean(episode_state)),
                }
            )
        vec_env.close()
    return pd.DataFrame(rows)


def rolling_mean(values: pd.Series, window: int = 3) -> np.ndarray:
    array = values.to_numpy(dtype=np.float64)
    out = np.zeros_like(array)
    for idx in range(array.size):
        left = max(0, idx - window + 1)
        out[idx] = array[left : idx + 1].mean()
    return out


def plot_big_figure(
    scope: str,
    plots_dir: pathlib.Path,
    summary_df: pd.DataFrame,
    training_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    force_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    qp_diag_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))

    for method, group in training_df.groupby("method"):
        axes[0, 0].plot(group["cumulative_timesteps"], group["rolling_episode_return"], label=method, linewidth=2.0)
    axes[0, 0].set_title(f"{scope}: training return")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Episode return")
    axes[0, 0].grid(alpha=0.3)

    for ax, df, title in [
        (axes[0, 1], clean_df, "clean eval mean±std"),
        (axes[0, 2], force_df, "force adversarial eval mean±std"),
        (axes[1, 0], control_df, "control-proxy eval mean±std"),
    ]:
        for method, group in df.groupby("method"):
            ax.plot(group["timesteps"], group["rolling_mean_reward"], linewidth=2.0, label=method)
            ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], alpha=0.12)
        ax.set_title(f"{scope}: {title}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean reward")
        ax.grid(alpha=0.3)

    for method, group in metrics_df.groupby("method"):
        axes[1, 1].plot(group["num_timesteps"], group["actor_update_norm"], linewidth=2.0, label=method)
        axes[1, 2].plot(group["num_timesteps"], group["critic_update_norm"], linewidth=2.0, label=method)
    axes[1, 1].set_title(f"{scope}: actor update norm")
    axes[1, 2].set_title(f"{scope}: critic update norm")
    for ax in [axes[1, 1], axes[1, 2]]:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Update norm")
        ax.grid(alpha=0.3)

    if not qp_diag_df.empty:
        qp_group = qp_diag_df[qp_diag_df["method"] == "proposed_qp"].copy()
        noG_group = qp_diag_df[qp_diag_df["method"] == "proposed_noG"].copy()
        for label, group in [("proposed_qp_beta", qp_group), ("proposed_qp_gamma", qp_group)]:
            pass
        if not qp_group.empty:
            axes[2, 0].plot(qp_group["step_index"], qp_group["beta"], label="beta", linewidth=2.0)
            axes[2, 0].plot(qp_group["step_index"], qp_group["gamma"], label="gamma", linewidth=2.0)
            axes[2, 1].plot(qp_group["step_index"], qp_group["gamma_active_frac"], label="gamma_active_frac", linewidth=2.0)
            axes[2, 1].plot(qp_group["step_index"], qp_group["G_contribution_norm"], label="G_contribution_norm", linewidth=2.0)
        if not noG_group.empty:
            axes[2, 0].plot(noG_group["step_index"], noG_group["beta"], label="noG_beta", linewidth=1.3, linestyle="--")
        axes[2, 0].set_title(f"{scope}: proposed beta/gamma vs updates")
        axes[2, 1].set_title(f"{scope}: proposed gamma_active_frac / G_contribution_norm")
        for ax in [axes[2, 0], axes[2, 1]]:
            ax.set_xlabel("Update index")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    bar_methods = summary_df["method"].tolist()
    x = np.arange(len(bar_methods))
    width = 0.25
    axes[2, 2].bar(x - width, summary_df["last5_clean_mean"], width=width, label="clean")
    axes[2, 2].bar(x, summary_df["last5_adversarial_mean"], width=width, label="force")
    axes[2, 2].bar(x + width, summary_df["last5_control_proxy_mean"], width=width, label="control-proxy")
    axes[2, 2].set_xticks(x)
    axes[2, 2].set_xticklabels(bar_methods, rotation=35)
    axes[2, 2].set_title(f"{scope}: final last5 comparison")
    axes[2, 2].grid(alpha=0.3)
    axes[2, 2].legend(fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
    fig.suptitle(f"scope = {scope} | deterministic=True | n_eval_episodes=20", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plots_dir / f"{scope}_all_methods_big.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def monotonic_nonincreasing(values: List[float], tol: float = 1e-6) -> bool:
    return all(values[idx + 1] <= values[idx] + tol for idx in range(len(values) - 1))


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    runs_dir = output_root / f"{args.scope}_runs_seed{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(pathlib.Path(args.stage5_root), pathlib.Path(args.stage8c_root), args.scope)

    summary_rows = []
    training_frames = []
    clean_frames = []
    force_frames = []
    control_frames = []
    metrics_frames = []
    qp_diag_frames = []
    sweep_frames = []

    for candidate in candidates:
        run_dir = ensure_run(candidate, args, runs_dir)
        analysis_dir = runs_dir / candidate.method / "analysis"
        run_summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        run_summary["scope"] = args.scope
        summary_rows.append(run_summary)

        train_df = pd.read_csv(analysis_dir / "training_episode_returns.csv")
        train_df["rolling_episode_return"] = rolling_mean(train_df["episode_return"], window=15)
        training_frames.append(train_df)

        clean_df = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
        force_df = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
        control_df = pd.read_csv(analysis_dir / "control_proxy_eval_returns.csv")
        for df in [clean_df, force_df, control_df]:
            df["scope"] = args.scope
            df["rolling_mean_reward"] = rolling_mean(df["mean_reward"])
        clean_frames.append(clean_df)
        force_frames.append(force_df)
        control_frames.append(control_df)

        metrics_path = run_dir / "analysis" / "protagonist_training_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            metrics_df["method"] = candidate.method
            metrics_df["scope"] = args.scope
            metrics_frames.append(metrics_df)

        for diag_file in run_dir.glob("protagonist_*diagnostics.csv"):
            diag_df = pd.read_csv(diag_file)
            diag_df["method"] = candidate.method
            diag_df["scope"] = args.scope
            qp_diag_frames.append(diag_df)

        sweep_frames.append(evaluate_strength_sweep(run_dir, candidate.method, args.scope, args.device, args.n_eval_episodes))

    summary_df = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    training_all = pd.concat(training_frames, ignore_index=True)
    clean_all = pd.concat(clean_frames, ignore_index=True)
    force_all = pd.concat(force_frames, ignore_index=True)
    control_all = pd.concat(control_frames, ignore_index=True)
    metrics_all = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    qp_diag_all = pd.concat(qp_diag_frames, ignore_index=True) if qp_diag_frames else pd.DataFrame()
    sweep_all = pd.concat(sweep_frames, ignore_index=True)

    summary_df.to_csv(output_root / f"{args.scope}_all_methods_summary.csv", index=False)
    training_all.to_csv(output_root / f"{args.scope}_training_curves.csv", index=False)
    clean_all.to_csv(output_root / f"{args.scope}_clean_eval_curves.csv", index=False)
    force_all.to_csv(output_root / f"{args.scope}_force_adv_eval_curves.csv", index=False)
    control_all.to_csv(output_root / f"{args.scope}_control_proxy_eval_curves.csv", index=False)
    metrics_all.to_csv(output_root / f"{args.scope}_update_diagnostics.csv", index=False)
    qp_diag_all.to_csv(output_root / f"{args.scope}_qp_diagnostics.csv", index=False)
    sweep_all.to_csv(output_root / f"{args.scope}_adv_strength_sweeps.csv", index=False)

    plot_big_figure(args.scope, plots_dir, summary_df, training_all, clean_all, force_all, control_all, metrics_all, qp_diag_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for adv_impact, ax in [("force", axes[0]), ("control-proxy", axes[1])]:
        subset = sweep_all[sweep_all["adv_impact"] == adv_impact]
        for method, group in subset.groupby("method"):
            ax.plot(group["adv_strength"], group["mean_return"], marker="o", linewidth=2.0, label=method)
        ax.set_title(f"{args.scope}: {adv_impact} strength sweep")
        ax.set_xlabel("Adversary strength")
        ax.set_ylabel("Mean return")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{args.scope}_strength_sweeps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    proposed_qp_row = summary_df[summary_df["method"] == "proposed_qp"].iloc[0]
    proposed_nog_row = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
    egm_row = summary_df[summary_df["method"] == "egm"].iloc[0]
    ppm_row = summary_df[summary_df["method"] == "ppm"].iloc[0]
    sgd_row = summary_df[summary_df["method"] == "sgd"].iloc[0]
    qp_subset = qp_diag_all[qp_diag_all["method"] == "proposed_qp"]
    gamma_healthy = (not qp_subset.empty) and float(qp_subset["gamma_active_frac"].mean()) > 0.0 and float(qp_subset["zero_update_flag"].mean()) < 0.5
    force_sweep = sweep_all[sweep_all["adv_impact"] == "force"]
    control_sweep = sweep_all[sweep_all["adv_impact"] == "control-proxy"]
    force_monotonic = all(
        monotonic_nonincreasing(group.sort_values("adv_strength")["mean_return"].tolist())
        for _, group in force_sweep.groupby("method")
    )
    control_more_sensitive = float(control_sweep.groupby("method")["mean_return"].min().mean()) < float(force_sweep.groupby("method")["mean_return"].min().mean())

    report_lines = [
        f"# {args.scope} All-Methods Report",
        "",
        "- Training protocol: `force`",
        "- Eval protocols: `clean`, `force adversarial`, `control-proxy`",
        "- deterministic: `True`",
        "- n_eval_episodes: `20`",
        "",
        "## Method summary",
        "",
    ]
    for _, row in summary_df.iterrows():
        report_lines.extend(
            [
                f"- `{row['method']}`",
                f"  - clean last5 mean: `{float(row['last5_clean_mean']):.6f}`",
                f"  - force-adv last5 mean: `{float(row['last5_adversarial_mean']):.6f}`",
                f"  - control-proxy last5 mean: `{float(row['last5_control_proxy_mean']):.6f}`",
                f"  - crash/nan: `{int(row['crash_flag'])}` / `{int(row['nan_flag'])}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- proposed_qp > proposed_noG on clean last5 mean: `{float(proposed_qp_row['last5_clean_mean']) > float(proposed_nog_row['last5_clean_mean'])}`",
            f"- proposed_qp > proposed_noG on force-adv last5 mean: `{float(proposed_qp_row['last5_adversarial_mean']) > float(proposed_nog_row['last5_adversarial_mean'])}`",
            f"- proposed_qp > EGM on clean last5 mean: `{float(proposed_qp_row['last5_clean_mean']) > float(egm_row['last5_clean_mean'])}`",
            f"- proposed_qp > PPM on clean last5 mean: `{float(proposed_qp_row['last5_clean_mean']) > float(ppm_row['last5_clean_mean'])}`",
            f"- proposed_qp > SGD on clean last5 mean: `{float(proposed_qp_row['last5_clean_mean']) > float(sgd_row['last5_clean_mean'])}`",
            f"- proposed_qp gamma/beta diagnostics healthy: `{gamma_healthy}`",
            f"- {args.scope} scope stable (no crash/no NaN across methods): `{bool((summary_df['crash_flag'] == 0).all() and (summary_df['nan_flag'] == 0).all())}`",
            "",
            "## Protocol readout",
            "",
            f"- force sweep monotonic damage across methods: `{force_monotonic}`",
            f"- control-proxy sweep more sensitive than force on average: `{control_more_sensitive}`",
            f"- force suitable as paper main robustness protocol: `{force_monotonic}`",
            f"- proposed_qp healthy enough to enter proper control-RARL: `{gamma_healthy and float(proposed_qp_row['last5_clean_mean']) >= float(proposed_nog_row['last5_clean_mean'])}`",
        ]
    )
    (output_root / f"{args.scope}_all_methods_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
