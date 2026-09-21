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


@dataclass(frozen=True)
class Candidate:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 7E short proposed-QP integration comparison")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--stage7d-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--N-mu", type=int, default=None)
    parser.add_argument("--N-nu", type=int, default=None)
    parser.add_argument("--adv-delay", type=int, default=-1)
    parser.add_argument(
        "--rarl-update-mode",
        choices=["alternating", "synchronized_lagged"],
        default="alternating",
    )
    parser.add_argument("--clean-warmup", action="store_true", default=False)
    parser.add_argument("--composite-outer-mode", choices=["none", "diagnostic", "gda"], default="none")
    parser.add_argument("--composite-outer-warmup", type=int, default=10)
    parser.add_argument("--composite-critic-updates", type=int, default=100)
    parser.add_argument("--composite-batch-size", type=int, default=256)
    parser.add_argument("--composite-state-batch-size", type=int, default=256)
    parser.add_argument("--composite-actor-lr", type=float, default=1e-6)
    parser.add_argument("--adv-fraction", type=float, default=None)
    parser.add_argument(
        "--adv-index-list",
        nargs="+",
        default=None,
        help="MuJoCo body names receiving adversarial forces (for example, torso or pole).",
    )
    parser.add_argument("--adv-impact", choices=("force", "control"), default="force")
    parser.add_argument("--control-proxy-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shared-lr", type=float, default=None)
    parser.add_argument("--shared-max-grad-norm", type=float, default=None)
    parser.add_argument("--shared-vf-coef", type=float, default=None)
    parser.add_argument("--qp-g-alpha", type=float, default=1e-3)
    parser.add_argument("--qp-eps", type=float, default=1e-8)
    parser.add_argument("--max-update-norm", type=float, default=float("inf"))
    parser.add_argument(
        "--perflyap-scope",
        choices=[
            "actor_mean_only",
            "actor_game",
            "actor_mean_heavy",
            "critic_downweighted",
            "logstd_excluded",
            "full_policy_actor_weighted",
            "actor_mean_plus_adv_actor_mean",
        ],
        default="full_policy_actor_weighted",
    )
    parser.add_argument("--perflyap-lambda-n", type=float, default=0.1)
    parser.add_argument("--perflyap-lambda-p", type=float, default=1.0)
    parser.add_argument("--perflyap-lambda-critic", type=float, default=0.1)
    parser.add_argument("--perflyap-logstd-weight", type=float, default=0.0)
    parser.add_argument("--perflyap-beta-max", type=float, default=1e-2)
    parser.add_argument("--perflyap-gamma-max", type=float, default=3e-5)
    parser.add_argument("--perflyap-update-cap", type=float, default=3e-3)
    parser.add_argument("--perflyap-fd-eps", type=float, default=1e-3)
    parser.add_argument("--perflyap-beta-probe", type=float, default=1e-3)
    parser.add_argument("--perflyap-gamma-probe", type=float, default=1e-6)
    parser.add_argument("--perflyap-ridge", type=float, default=1e-8)
    parser.add_argument("--perflyap-rho", type=float, default=1e-8)
    parser.add_argument("--perflyap-scale-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--perflyap-cost-mode",
        choices=[
            "actor_surrogate_cost",
            "unclipped_actor_surrogate_cost",
            "mixed_clean_unclipped_actor_surrogate_cost",
            "mixed_rarl_unclipped_actor_surrogate_cost",
        ],
        default="unclipped_actor_surrogate_cost",
    )
    parser.add_argument(
        "--perflyap-selector-mode",
        choices=[
            "current_q_pred",
            "fixed_nog",
            "actual_surrogate_selector",
            "actual_mixed_selector",
            "ls_capaware",
            "safe_fixed_minusg",
        ],
        default="current_q_pred",
    )
    parser.add_argument(
        "--perflyap-direction-mode",
        choices=["egm_plus_JF_F", "egm_minus_JF_F", "performance_grad"],
        default="egm_plus_JF_F",
    )
    parser.add_argument(
        "--ppm-inner-steps",
        type=int,
        default=None,
        help="Optional PPM fixed-point iteration override. Use >2 for a baseline distinct from EGM.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["adam", "sgd", "egm", "ppm"],
        choices=[
            "adam",
            "sgd",
            "egm",
            "ppm",
            "proposed_noG",
            "proposed_qp",
            "surrogate_noG",
            "surrogate_qp",
        ],
        help="Subset of methods to run. Single-method selection is supported for HPC arrays.",
    )
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def find_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def load_baseline_candidates(stage5_root: pathlib.Path, stage7d_root: pathlib.Path, args: argparse.Namespace) -> List[Candidate]:
    config_map = json.loads((stage5_root / "final_full_policy_optimizer_configs.json").read_text(encoding="utf-8"))
    # Keep this input in the interface for compatibility with existing launchers.
    # The performance-aligned composite optimizer does not use the legacy QP normalization.

    methods = ["adam", "sgd", "egm", "ppm"]
    candidates: List[Candidate] = []
    for method in methods:
        spec = config_map[method]
        optimizer_kwargs = dict(spec.get("optimizer_kwargs", {}))
        if method == "ppm" and args.ppm_inner_steps is not None:
            if args.ppm_inner_steps < 1:
                raise ValueError("--ppm-inner-steps must be >= 1")
            optimizer_kwargs["inner_steps"] = args.ppm_inner_steps
        candidates.append(
            Candidate(
                method=method,
                optimizer=str(spec["optimizer"]),
                lr=float(args.shared_lr if args.shared_lr is not None else spec["lr"]),
                max_grad_norm=float(args.shared_max_grad_norm if args.shared_max_grad_norm is not None else spec["max_grad_norm"]),
                vf_coef=float(args.shared_vf_coef if args.shared_vf_coef is not None else spec["vf_coef"]),
                optimizer_kwargs=optimizer_kwargs,
            )
        )

    egm_spec = config_map["egm"]
    composite_kwargs = {
        "perflyap_scope": args.perflyap_scope,
        "lambda_N": args.perflyap_lambda_n,
        "lambda_P": args.perflyap_lambda_p,
        "lambda_critic": args.perflyap_lambda_critic,
        "logstd_weight": args.perflyap_logstd_weight,
        "qp_fd_eps": args.perflyap_fd_eps,
        "qp_beta_probe": args.perflyap_beta_probe,
        "qp_gamma_probe": args.perflyap_gamma_probe,
        "qp_ridge": args.perflyap_ridge,
        "qp_rho": args.perflyap_rho,
        "qp_beta_max": args.perflyap_beta_max,
        "qp_gamma_max": args.perflyap_gamma_max,
        "qp_max_update_norm": args.perflyap_update_cap,
        "qp_eps": args.qp_eps,
        "eta_egm_reference": float(args.shared_lr if args.shared_lr is not None else egm_spec["lr"]),
        "use_scale_normalization": args.perflyap_scale_normalization,
        "cost_mode": args.perflyap_cost_mode,
        "selector_mode": args.perflyap_selector_mode,
        "direction_mode": args.perflyap_direction_mode,
    }
    for method_name, optimizer_name in [
        ("surrogate_noG", "proposed_noG_perfLyap"),
        ("surrogate_qp", "proposed_qp_perfLyap"),
    ]:
        candidates.append(
            Candidate(
                method=method_name,
                optimizer=optimizer_name,
                lr=float(args.shared_lr if args.shared_lr is not None else egm_spec["lr"]),
                max_grad_norm=float(
                    args.shared_max_grad_norm
                    if args.shared_max_grad_norm is not None
                    else egm_spec["max_grad_norm"]
                ),
                vf_coef=float(args.shared_vf_coef if args.shared_vf_coef is not None else egm_spec["vf_coef"]),
                optimizer_kwargs=dict(composite_kwargs),
            )
        )
    selected = set(args.methods)
    if selected.intersection({"proposed_noG", "proposed_qp"}):
        raise ValueError(
            "The paper-exact proposed optimizer requires field energy plus a finite-inner-step "
            "proximal saddle gap. It is not implemented in this detached PPO runner. Use the "
            "explicit surrogate_noG/surrogate_qp diagnostics only for PPO-surrogate audits."
        )
    if selected.intersection({"surrogate_noG", "surrogate_qp"}):
        if args.perflyap_lambda_n <= 0.0 or args.perflyap_lambda_p <= 0.0:
            raise ValueError(
                "Composite PerfLyap runs require --perflyap-lambda-n > 0 and "
                "--perflyap-lambda-p > 0; zero would select a non-composite special case."
            )
    candidates = [candidate for candidate in candidates if candidate.method in selected]
    return candidates


def run_candidate(args: argparse.Namespace, candidate: Candidate, output_root: pathlib.Path) -> pathlib.Path:
    run_root = output_root / candidate.method
    saved_models_dir = run_root / "saved_models"
    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "resolved_optimizer_manifest.json").write_text(
        json.dumps(
            {
                "method": candidate.method,
                "optimizer": candidate.optimizer,
                "lr": candidate.lr,
                "max_grad_norm": candidate.max_grad_norm,
                "vf_coef": candidate.vf_coef,
                "optimizer_kwargs": candidate.optimizer_kwargs,
                "lyapunov_implementation": (
                    "ppo_surrogate_composite_perflyap"
                    if candidate.optimizer in {"proposed_noG_perfLyap", "proposed_qp_perfLyap"}
                    else "none"
                ),
                "paper_exact_proximal_gap": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    analysis_dir = run_root / "analysis"
    latest_run_dir = None
    env_root = saved_models_dir / "rarl-ppo" / args.env
    if env_root.exists():
        latest_run_dir = find_run_dir(saved_models_dir, args.env)
    control_proxy_npz = None if latest_run_dir is None else latest_run_dir / "control_proxy_eval" / "evaluations.npz"
    if latest_run_dir is not None and control_proxy_npz is not None and control_proxy_npz.exists():
        if (analysis_dir / "run_summary.csv").exists() and (analysis_dir / "control_proxy_eval_returns.csv").exists():
            return latest_run_dir
        analysis_dir.mkdir(parents=True, exist_ok=True)
        if (run_root / "stdout.txt").exists():
            shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
        if (run_root / "stderr.txt").exists():
            shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
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
        args.adv_impact,
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
        "--adv-delay",
        str(args.adv_delay),
        "--rarl-update-mode",
        args.rarl_update_mode,
        "--saved-models-path",
        str(saved_models_dir),
        "--log-folder",
        str(logging_dir),
        "--tensorboard-log",
        str(tb_dir),
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
        "--composite-outer-mode",
        args.composite_outer_mode,
        "--composite-outer-warmup",
        str(args.composite_outer_warmup),
        "--composite-critic-updates",
        str(args.composite_critic_updates),
        "--composite-batch-size",
        str(args.composite_batch_size),
        "--composite-state-batch-size",
        str(args.composite_state_batch_size),
        "--composite-actor-lr",
        str(args.composite_actor_lr),
    ]
    if args.control_proxy_eval:
        command.append("--control-proxy-eval")
    if args.N_mu is not None:
        command.extend(["--N-mu", str(args.N_mu)])
    if args.N_nu is not None:
        command.extend(["--N-nu", str(args.N_nu)])
    if args.clean_warmup:
        command.append("--clean-warmup")
    if args.adv_fraction is not None:
        command.extend(["--adv-fraction", str(args.adv_fraction)])
    if args.adv_index_list is not None:
        command.extend(["--adv-index-list", *args.adv_index_list])
    protagonist_kwargs_tokens = []
    adversary_kwargs_tokens = []
    for key, value in sorted(candidate.optimizer_kwargs.items()):
        if isinstance(value, float) and not np.isfinite(value):
            continue
        rendered_value = repr(value) if isinstance(value, str) else value
        protagonist_kwargs_tokens.append(f"{key}:{rendered_value}")
        adversary_kwargs_tokens.append(f"{key}:{rendered_value}")
    if protagonist_kwargs_tokens:
        command.extend(["--protagonist-optimizer-kwargs", *protagonist_kwargs_tokens])
    if adversary_kwargs_tokens:
        command.extend(["--adversary-optimizer-kwargs", *adversary_kwargs_tokens])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
    run_dir = find_run_dir(saved_models_dir, args.env)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(run_dir), "--output-dir", str(analysis_dir), "--method", candidate.method],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return run_dir


def parse_eval_npz(path: pathlib.Path, method: str, eval_type: str) -> pd.DataFrame:
    data = np.load(path)
    rewards = data["results"]
    return pd.DataFrame(
        {
            "method": method,
            "timesteps": data["timesteps"],
            "mean_reward": rewards.mean(axis=1),
            "std_reward": rewards.std(axis=1),
            "min_reward": rewards.min(axis=1),
            "max_reward": rewards.max(axis=1),
            "mean_ep_length": data["ep_lengths"].mean(axis=1),
            "eval_type": eval_type,
        }
    )


def read_diagnostics(run_dir: pathlib.Path, method: str) -> pd.DataFrame:
    frames = []
    for path in run_dir.glob("*proposed*_diagnostics.csv"):
        df = pd.read_csv(path)
        df["method"] = method
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rolling_mean(values: pd.Series, window: int = 3) -> np.ndarray:
    array = values.to_numpy(dtype=np.float64)
    out = np.zeros_like(array)
    for idx in range(array.size):
        left = max(0, idx - window + 1)
        out[idx] = array[left : idx + 1].mean()
    return out


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    runs_dir = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_baseline_candidates(pathlib.Path(args.stage5_root), pathlib.Path(args.stage7d_root), args)

    summary_rows = []
    clean_frames = []
    adv_force_frames = []
    control_frames = []
    training_frames = []
    diagnostics_frames = []

    for candidate in candidates:
        run_dir = run_candidate(args, candidate, runs_dir)
        analysis_dir = runs_dir / candidate.method / "analysis"
        summary_rows.append(pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict())
        clean_df = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
        adv_df = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
        train_df = pd.read_csv(analysis_dir / "training_episode_returns.csv")
        control_npz = run_dir / "control_proxy_eval" / "evaluations.npz"
        if control_npz.exists():
            control_df = parse_eval_npz(control_npz, candidate.method, "control-proxy")
        else:
            control_df = adv_df.copy()
            control_df["eval_type"] = "control-proxy-not-applicable"
        clean_df["rolling_mean_reward"] = rolling_mean(clean_df["mean_reward"])
        adv_df["rolling_mean_reward"] = rolling_mean(adv_df["mean_reward"])
        control_df["rolling_mean_reward"] = rolling_mean(control_df["mean_reward"])
        train_df["rolling_episode_return"] = rolling_mean(train_df["episode_return"], window=15)
        clean_frames.append(clean_df)
        adv_force_frames.append(adv_df)
        control_frames.append(control_df)
        training_frames.append(train_df)
        diagnostics_df = read_diagnostics(run_dir, candidate.method)
        if not diagnostics_df.empty:
            diagnostics_frames.append(diagnostics_df)

    summary_df = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    clean_all = pd.concat(clean_frames, ignore_index=True)
    adv_force_all = pd.concat(adv_force_frames, ignore_index=True)
    control_all = pd.concat(control_frames, ignore_index=True)
    training_all = pd.concat(training_frames, ignore_index=True)
    diagnostics_all = pd.concat(diagnostics_frames, ignore_index=True) if diagnostics_frames else pd.DataFrame()

    control_summary = control_all.groupby("method").agg(
        final_control_proxy_return=("mean_reward", "last"),
        best_control_proxy_return=("mean_reward", "max"),
        last5_control_proxy_mean=("mean_reward", lambda values: float(pd.Series(values).tail(min(5, len(values))).mean())),
    ).reset_index()
    summary_df = summary_df.merge(control_summary, on="method", how="left", suffixes=("_existing", ""))
    for column in ("final_control_proxy_return", "best_control_proxy_return", "last5_control_proxy_mean"):
        existing_column = f"{column}_existing"
        if existing_column in summary_df.columns:
            summary_df[column] = summary_df[column].fillna(summary_df[existing_column])
            summary_df = summary_df.drop(columns=[existing_column])

    summary_df.to_csv(output_root / "stage7e_short_summary.csv", index=False)
    clean_all.to_csv(output_root / "short_clean_eval_all_methods.csv", index=False)
    adv_force_all.to_csv(output_root / "short_adv_eval_force_all_methods.csv", index=False)
    control_all.to_csv(output_root / "short_adv_eval_control_all_methods.csv", index=False)
    training_all.to_csv(output_root / "short_training_return_all_methods.csv", index=False)
    if not diagnostics_all.empty:
        diagnostics_all.to_csv(output_root / "proposed_qp_diagnostics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in clean_all.groupby("method"):
        ax.plot(group["timesteps"], group["rolling_mean_reward"], linewidth=2.0, label=method)
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], alpha=0.12)
    ax.set_title("Short Clean Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "short_clean_eval_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in adv_force_all.groupby("method"):
        ax.plot(group["timesteps"], group["rolling_mean_reward"], linewidth=2.0, label=method)
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], alpha=0.12)
    ax.set_title("Short Adversarial Eval (force)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "short_adv_eval_force_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in control_all.groupby("method"):
        ax.plot(group["timesteps"], group["rolling_mean_reward"], linewidth=2.0, label=method)
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], alpha=0.12)
    ax.set_title("Short Adversarial Eval (control-proxy)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "short_adv_eval_control_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in training_all.groupby("method"):
        ax.plot(group["cumulative_timesteps"], group["rolling_episode_return"], linewidth=2.0, label=method)
    ax.set_title("Short Training Return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "short_training_return_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not diagnostics_all.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for method, group in diagnostics_all.groupby("method"):
            grouped = group.groupby(group.index).mean(numeric_only=True)
            axes[0].plot(group.index, group["beta"], label=method, alpha=0.8)
            axes[1].plot(group.index, group["gamma"], label=method, alpha=0.8)
            axes[2].plot(group.index, group["update_norm_post_cap"], label=method, alpha=0.8)
        axes[0].set_title("Beta")
        axes[1].set_title("Gamma")
        axes[2].set_title("Update Norm")
        for axis in axes:
            axis.grid(alpha=0.3)
            axis.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "proposed_qp_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    report_lines = [
        "# Stage 7E Short Training Comparison",
        "",
        "- Training protocol: `force`",
        "- Additional eval protocol: `control-proxy` (sanity only, not control-trained adversary)",
        "",
        "## Final rows",
        "",
    ]
    for _, row in summary_df.iterrows():
        report_lines.extend(
            [
                f"- `{row['method']}`",
                f"  - clean last5 mean: `{row['last5_clean_mean']:.6f}`",
                f"  - force-adv last5 mean: `{row['last5_adversarial_mean']:.6f}`",
                f"  - control-proxy last5 mean: `{row['last5_control_proxy_mean']:.6f}`",
                f"  - crash/nan: `{int(row['crash_flag'])}` / `{int(row['nan_flag'])}`",
            ]
        )
    method_names = set(summary_df["method"])
    if {"proposed_qp", "proposed_noG"}.issubset(method_names):
        proposed_qp_row = summary_df[summary_df["method"] == "proposed_qp"].iloc[0]
        proposed_nog_row = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
        report_lines.extend(
            [
                "",
                "## Gate summary",
                "",
                f"- proposed_qp not worse than proposed_noG on clean last5 mean: `{float(proposed_qp_row['last5_clean_mean']) >= float(proposed_nog_row['last5_clean_mean'])}`",
                f"- proposed_qp not worse than proposed_noG on force-adv last5 mean: `{float(proposed_qp_row['last5_adversarial_mean']) >= float(proposed_nog_row['last5_adversarial_mean'])}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Note",
            "",
            "- If force-adv curves remain weakly harmful while control-proxy is strongly harmful, the force-only protocol should not be treated as the final paper robustness result.",
        ]
    )
    (output_root / "stage7e_short_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
