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
    parser = argparse.ArgumentParser("Stage 9 proper control-RARL benchmark")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--stage8c-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--scope", type=str, default="full_policy", choices=["full_policy", "actor_game"])
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
    adv_npz = None if latest_run_dir is None else latest_run_dir / "adv_eval" / "evaluations.npz"
    if latest_run_dir is not None and adv_npz is not None and adv_npz.exists():
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


def evaluate_control_strength_sweep(run_dir: pathlib.Path, method: str, scope: str, device: str, n_eval_episodes: int) -> pd.DataFrame:
    config_dir = next(path for path in run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists())
    saved_run = SavedRun(
        method=method,
        tag=method,
        run_root=run_dir,
        model_dir=config_dir,
        args_data=load_yaml(config_dir / "args.yml"),
        config_data=load_yaml(config_dir / "config.yml"),
    )
    strengths = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    rows = []
    model, vec_env = load_rarl_for_eval(saved_run, adv_impact="control", adv_strength=1.0, device=device)
    for strength in strengths:
        set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=strength)
        episode_rewards = []
        protagonist_action_norms = []
        adv_pre_norms = []
        adv_post_norms = []
        perturbation_norms = []
        clip_fractions = []
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_pro_norm = []
        ep_adv_pre = []
        ep_adv_post = []
        ep_perturb = []
        ep_clip = []
        while len(episode_rewards) < n_eval_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            ep_reward += float(rewards[0])
            info = infos[0]
            ep_pro_norm.append(float(info.get("protagonist_action_norm", 0.0)))
            ep_adv_pre.append(float(info.get("adversary_action_norm_pre_clip", 0.0)))
            ep_adv_post.append(float(info.get("adversary_action_norm_post_clip", 0.0)))
            ep_perturb.append(float(info.get("applied_control_perturbation_norm", info.get("applied_disturbance_norm", 0.0))))
            ep_clip.append(float(info.get("action_clip_fraction", 0.0)))
            if bool(dones[0]):
                episode_rewards.append(ep_reward)
                protagonist_action_norms.append(float(np.mean(ep_pro_norm)) if ep_pro_norm else 0.0)
                adv_pre_norms.append(float(np.mean(ep_adv_pre)) if ep_adv_pre else 0.0)
                adv_post_norms.append(float(np.mean(ep_adv_post)) if ep_adv_post else 0.0)
                perturbation_norms.append(float(np.mean(ep_perturb)) if ep_perturb else 0.0)
                clip_fractions.append(float(np.mean(ep_clip)) if ep_clip else 0.0)
                obs = vec_env.reset()
                ep_reward = 0.0
                ep_pro_norm = []
                ep_adv_pre = []
                ep_adv_post = []
                ep_perturb = []
                ep_clip = []
        rows.append(
            {
                "scope": scope,
                "method": method,
                "adv_impact": "control",
                "adv_strength": strength,
                "mean_return": float(np.mean(episode_rewards)),
                "std_return": float(np.std(episode_rewards)),
                "protagonist_action_norm": float(np.mean(protagonist_action_norms)),
                "adversary_action_norm_pre_clip": float(np.mean(adv_pre_norms)),
                "adversary_action_norm_post_clip": float(np.mean(adv_post_norms)),
                "applied_control_perturbation_norm": float(np.mean(perturbation_norms)),
                "action_clip_fraction": float(np.mean(clip_fractions)),
            }
        )
    vec_env.close()
    return pd.DataFrame(rows)


def load_analysis_frame(analysis_dir: pathlib.Path, filename: str, method: str) -> pd.DataFrame:
    primary = analysis_dir / filename
    if primary.exists():
        frame = pd.read_csv(primary)
    elif filename == "training_returns.csv":
        fallback = analysis_dir / "training_episode_returns.csv"
        if not fallback.exists():
            raise FileNotFoundError(f"Neither {primary} nor {fallback} exists")
        frame = pd.read_csv(fallback)
    else:
        raise FileNotFoundError(f"Missing analysis file: {primary}")
    frame["method"] = method
    return frame


def build_summary_row(run_summary: pd.Series, candidate: Candidate, scope: str) -> Dict[str, object]:
    row = run_summary.to_dict()
    row["optimizer_label"] = candidate.method
    row["protagonist_optimizer"] = candidate.optimizer
    row["adversary_optimizer"] = candidate.optimizer
    row["ppm_inner_steps"] = int(candidate.optimizer_kwargs.get("inner_steps", -1))
    row["scope"] = scope
    return row


def save_big_figure(
    *,
    training_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    plot_path: pathlib.Path,
    title: str,
) -> None:
    train_x = "timesteps" if "timesteps" in training_df.columns else "cumulative_timesteps"
    train_y = "episode_return" if "episode_return" in training_df.columns else "mean_episode_return"
    eval_y = "mean_return" if "mean_return" in clean_df.columns else "mean_reward"
    eval_std = "std_return" if "std_return" in clean_df.columns else "std_reward"
    methods = list(summary_df["method"])
    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
        "proposed_noG": "tab:purple",
        "proposed_qp": "tab:blue",
    }
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    ax = axes[0, 0]
    for method in methods:
        sub = training_df[training_df["method"] == method]
        ax.plot(sub[train_x], sub[train_y], label=method, color=colors.get(method))
    ax.set_title("Training Return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode return")
    ax.legend()

    ax = axes[0, 1]
    for method in methods:
        sub = clean_df[clean_df["method"] == method]
        ax.plot(sub["timesteps"], sub[eval_y], label=method, color=colors.get(method))
        if eval_std in sub.columns:
            ax.fill_between(sub["timesteps"], sub[eval_y] - sub[eval_std], sub[eval_y] + sub[eval_std], color=colors.get(method), alpha=0.15)
    ax.set_title("Clean Eval (deterministic=True, n=20)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean return")

    ax = axes[1, 0]
    for method in methods:
        sub = adv_df[adv_df["method"] == method]
        ax.plot(sub["timesteps"], sub[eval_y], label=method, color=colors.get(method))
        if eval_std in sub.columns:
            ax.fill_between(sub["timesteps"], sub[eval_y] - sub[eval_std], sub[eval_y] + sub[eval_std], color=colors.get(method), alpha=0.15)
    ax.set_title("Control Adversarial Eval (deterministic=True, n=20)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean return")

    ax = axes[1, 1]
    for method in methods:
        sub = sweep_df[sweep_df["method"] == method]
        ax.plot(sub["adv_strength"], sub["mean_return"], marker="o", label=method, color=colors.get(method))
    ax.set_title("Control Robustness Sweep")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Mean return")

    ax = axes[2, 0]
    for method in methods:
        sub = sweep_df[sweep_df["method"] == method]
        ax.plot(sub["adv_strength"], sub["applied_control_perturbation_norm"], marker="o", label=method, color=colors.get(method))
    ax.set_title("Applied Control Perturbation")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Perturbation norm")

    ax = axes[2, 1]
    bar_df = summary_df[["method", "last5_clean_mean", "last5_adversarial_mean"]].copy()
    x = np.arange(len(bar_df))
    width = 0.35
    ax.bar(x - width / 2, bar_df["last5_clean_mean"], width=width, label="clean")
    ax.bar(x + width / 2, bar_df["last5_adversarial_mean"], width=width, label="control-adv")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_df["method"], rotation=25, ha="right")
    ax.set_title("Final last5 means")
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    stage5_root = pathlib.Path(args.stage5_root)
    stage8c_root = pathlib.Path(args.stage8c_root)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(stage5_root, stage8c_root, args.scope)
    runs_dir = output_root / f"{args.scope}_runs_seed{args.seed}"

    summary_rows = []
    training_frames = []
    clean_frames = []
    adv_frames = []
    sweep_frames = []

    for candidate in candidates:
        latest_run_dir = ensure_run(candidate, args, runs_dir)
        analysis_dir = runs_dir / candidate.method / "analysis"
        run_summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0]
        summary_rows.append(build_summary_row(run_summary, candidate, args.scope))
        training_frames.append(load_analysis_frame(analysis_dir, "training_returns.csv", candidate.method))
        clean_frames.append(load_analysis_frame(analysis_dir, "clean_eval_returns.csv", candidate.method))
        adv_frames.append(load_analysis_frame(analysis_dir, "adversarial_eval_returns.csv", candidate.method))
        sweep_frames.append(evaluate_control_strength_sweep(latest_run_dir, candidate.method, args.scope, args.device, args.n_eval_episodes))

    summary_df = pd.DataFrame(summary_rows)
    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    sweep_df = pd.concat(sweep_frames, ignore_index=True)

    summary_df.to_csv(output_root / "control_training_summary.csv", index=False)
    clean_df.to_csv(output_root / "control_clean_eval_curves.csv", index=False)
    adv_df.to_csv(output_root / "control_adv_eval_curves.csv", index=False)
    sweep_df.to_csv(output_root / "control_robustness_sweep.csv", index=False)

    save_big_figure(
        training_df=training_df,
        clean_df=clean_df,
        adv_df=adv_df,
        sweep_df=sweep_df,
        summary_df=summary_df,
        plot_path=plots_dir / "control_all_methods_big.png",
        title=f"Proper control-RARL benchmark (scope={args.scope}, deterministic=True, n_eval_episodes={args.n_eval_episodes})",
    )

    plt.figure(figsize=(8, 5))
    for method, sub in sweep_df.groupby("method"):
        plt.plot(sub["adv_strength"], sub["mean_return"], marker="o", label=method)
    plt.title("Control robustness sweep")
    plt.xlabel("Adversary strength")
    plt.ylabel("Mean return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "control_robustness_sweep.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for method, sub in sweep_df.groupby("method"):
        plt.plot(sub["adv_strength"], sub["applied_control_perturbation_norm"], marker="o", label=method)
    plt.title("Applied control perturbation vs strength")
    plt.xlabel("Adversary strength")
    plt.ylabel("Perturbation norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "control_action_perturbation_vs_strength.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for method, sub in sweep_df.groupby("method"):
        plt.plot(sub["adv_strength"], sub["action_clip_fraction"], marker="o", label=method)
    plt.title("Control action clip fraction vs strength")
    plt.xlabel("Adversary strength")
    plt.ylabel("Clip fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "control_clip_fraction_vs_strength.png", dpi=200, bbox_inches="tight")
    plt.close()

    monotonic_flags = {}
    for method, sub in sweep_df.groupby("method"):
        ordered = sub.sort_values("adv_strength")
        diffs = np.diff(ordered["mean_return"].to_numpy(dtype=float))
        monotonic_flags[method] = bool(np.all(diffs <= 1e-6))

    qp_row = summary_df[summary_df["method"] == "proposed_qp"].iloc[0]
    nog_row = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
    egm_row = summary_df[summary_df["method"] == "egm"].iloc[0]
    ppm_row = summary_df[summary_df["method"] == "ppm"].iloc[0]
    sgd_row = summary_df[summary_df["method"] == "sgd"].iloc[0]

    lines = [
        f"# Proper control-RARL report ({args.scope})",
        "",
        f"- training adv-impact: `control`",
        f"- deterministic eval: `True`",
        f"- n_eval_episodes: `{args.n_eval_episodes}`",
        "",
        "## Key answers",
        "",
        f"- control adversary monotonic degradation for all methods: `{all(monotonic_flags.values())}`",
        f"- proposed_qp > proposed_noG on clean last5 mean: `{float(qp_row['last5_clean_mean']) > float(nog_row['last5_clean_mean'])}`",
        f"- proposed_qp > proposed_noG on control-adv last5 mean: `{float(qp_row['last5_adversarial_mean']) > float(nog_row['last5_adversarial_mean'])}`",
        f"- proposed_qp > EGM on clean last5 mean: `{float(qp_row['last5_clean_mean']) > float(egm_row['last5_clean_mean'])}`",
        f"- proposed_qp > PPM on clean last5 mean: `{float(qp_row['last5_clean_mean']) > float(ppm_row['last5_clean_mean'])}`",
        f"- proposed_qp > SGD on clean last5 mean: `{float(qp_row['last5_clean_mean']) > float(sgd_row['last5_clean_mean'])}`",
        f"- scope used for this Stage 9 run: `{args.scope}`",
        "",
        "## Per-method monotonicity",
        "",
    ]
    for method, flag in monotonic_flags.items():
        lines.append(f"- `{method}` monotonic under control strength sweep: `{flag}`")

    (output_root / "control_rarl_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
