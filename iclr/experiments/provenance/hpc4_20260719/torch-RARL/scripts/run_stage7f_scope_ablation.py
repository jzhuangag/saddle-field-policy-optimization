from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ScopeCandidate:
    method: str
    optimizer_scope: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 7F proposed-QP scope ablation")
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
    parser.add_argument("--qp-g-alpha", type=float, default=1e-3)
    parser.add_argument("--qp-eps", type=float, default=1e-8)
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def load_candidates(stage5_root: pathlib.Path, stage7d_root: pathlib.Path, args: argparse.Namespace) -> List[ScopeCandidate]:
    config_map = json.loads((stage5_root / "final_full_policy_optimizer_configs.json").read_text(encoding="utf-8"))
    qp_selection = json.loads((stage7d_root / "stage7d_selected_qp_normalization.json").read_text(encoding="utf-8"))
    normalization = qp_selection["qp_normalization"]
    egm_spec = config_map["egm"]
    common_kwargs = {
        "qp_normalization": normalization,
        "qp_g_alpha": args.qp_g_alpha,
        "qp_eps": args.qp_eps,
    }
    return [
        ScopeCandidate(
            method="proposed_qp_full_policy",
            optimizer_scope="full_policy",
            lr=float(egm_spec["lr"]),
            max_grad_norm=float(egm_spec["max_grad_norm"]),
            vf_coef=float(egm_spec["vf_coef"]),
            optimizer_kwargs={**common_kwargs, "optimizer_scope": "full_policy"},
        ),
        ScopeCandidate(
            method="proposed_qp_actor_game",
            optimizer_scope="actor_game",
            lr=float(egm_spec["lr"]),
            max_grad_norm=float(egm_spec["max_grad_norm"]),
            vf_coef=float(egm_spec["vf_coef"]),
            optimizer_kwargs={**common_kwargs, "optimizer_scope": "actor_game"},
        ),
    ]


def ensure_analysis(candidate: ScopeCandidate, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
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
        "--protagonist-optimizer",
        "proposed_qp",
        "--adversary-optimizer",
        "proposed_qp",
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
    protagonist_kwargs_tokens = []
    adversary_kwargs_tokens = []
    for key, value in sorted(candidate.optimizer_kwargs.items()):
        rendered_value = repr(value) if isinstance(value, str) else value
        protagonist_kwargs_tokens.append(f"{key}:{rendered_value}")
        adversary_kwargs_tokens.append(f"{key}:{rendered_value}")
    if protagonist_kwargs_tokens:
        command.extend(["--protagonist-optimizer-kwargs", *protagonist_kwargs_tokens])
    if adversary_kwargs_tokens:
        command.extend(["--adversary-optimizer-kwargs", *adversary_kwargs_tokens])

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


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    runs_dir = output_root / "scope_runs"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(pathlib.Path(args.stage5_root), pathlib.Path(args.stage7d_root), args)

    summaries = []
    clean_frames = []
    adv_frames = []
    control_frames = []
    diag_rows = []
    for candidate in candidates:
        run_dir = ensure_analysis(candidate, args, runs_dir)
        analysis_dir = runs_dir / candidate.method / "analysis"
        summaries.append(pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict())
        clean_df = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
        adv_df = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
        control_df = pd.read_csv(analysis_dir / "control_proxy_eval_returns.csv")
        clean_frames.append(clean_df)
        adv_frames.append(adv_df)
        control_frames.append(control_df)

        diag_path = run_dir / "protagonist_proposed_qp_diagnostics.csv"
        if diag_path.exists():
            diag_df = pd.read_csv(diag_path)
            gamma_active_col = "gamma_active_flag" if "gamma_active_flag" in diag_df.columns else "gamma_active"
            diag_rows.append(
                {
                    "method": candidate.method,
                    "optimizer_scope": candidate.optimizer_scope,
                    "beta_mean": float(diag_df["beta"].mean()),
                    "gamma_mean": float(diag_df["gamma"].mean()),
                    "gamma_active_frac": float(diag_df[gamma_active_col].mean()),
                    "zero_update_frac": float(diag_df["zero_update_flag"].mean()),
                    "boundary_frac": float(diag_df["boundary_solution_flag"].mean()),
                    "interior_frac": float(diag_df["interior_solution_flag"].mean()),
                    "update_norm_mean": float(diag_df["update_norm_post_cap"].mean()),
                }
            )

    summary_df = pd.DataFrame(summaries).sort_values("method").reset_index(drop=True)
    diag_df = pd.DataFrame(diag_rows)
    summary_df.to_csv(output_root / "scope_ablation_summary.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(output_root / "scope_ablation_diagnostics.csv", index=False)

    clean_all = pd.concat(clean_frames, ignore_index=True)
    adv_all = pd.concat(adv_frames, ignore_index=True)
    control_all = pd.concat(control_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for method, group in clean_all.groupby("method"):
        axes[0].plot(group["timesteps"], group["mean_reward"], linewidth=2.0, label=method)
    axes[0].set_title("Clean Eval")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for method, group in adv_all.groupby("method"):
        axes[1].plot(group["timesteps"], group["mean_reward"], linewidth=2.0, label=method)
    axes[1].set_title("Force Adversarial Eval")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Mean reward")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    for method, group in control_all.groupby("method"):
        axes[2].plot(group["timesteps"], group["mean_reward"], linewidth=2.0, label=method)
    axes[2].set_title("Control-Proxy Eval")
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Mean reward")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "proposed_qp_scope_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    full_row = summary_df[summary_df["method"] == "proposed_qp_full_policy"].iloc[0]
    actor_row = summary_df[summary_df["method"] == "proposed_qp_actor_game"].iloc[0]
    report_lines = [
        "# Stage 7F Scope Ablation",
        "",
        "- Training protocol: `force`",
        "- Additional eval protocol: `control-proxy` (sanity only, not control-trained adversary)",
        "",
        "## Summary",
        "",
        f"- `proposed_qp_full_policy` clean last5 mean: `{float(full_row['last5_clean_mean']):.6f}`",
        f"- `proposed_qp_full_policy` force-adv last5 mean: `{float(full_row['last5_adversarial_mean']):.6f}`",
        f"- `proposed_qp_full_policy` control-proxy last5 mean: `{float(full_row['last5_control_proxy_mean']):.6f}`",
        f"- `proposed_qp_actor_game` clean last5 mean: `{float(actor_row['last5_clean_mean']):.6f}`",
        f"- `proposed_qp_actor_game` force-adv last5 mean: `{float(actor_row['last5_adversarial_mean']):.6f}`",
        f"- `proposed_qp_actor_game` control-proxy last5 mean: `{float(actor_row['last5_control_proxy_mean']):.6f}`",
    ]
    if not diag_df.empty:
        full_diag = diag_df[diag_df["method"] == "proposed_qp_full_policy"].iloc[0]
        actor_diag = diag_df[diag_df["method"] == "proposed_qp_actor_game"].iloc[0]
        report_lines.extend(
            [
                "",
                "## Diagnostics",
                "",
                f"- `full_policy` gamma_active_frac: `{float(full_diag['gamma_active_frac']):.6f}`",
                f"- `actor_game` gamma_active_frac: `{float(actor_diag['gamma_active_frac']):.6f}`",
                f"- `full_policy` zero_update_frac: `{float(full_diag['zero_update_frac']):.6f}`",
                f"- `actor_game` zero_update_frac: `{float(actor_diag['zero_update_frac']):.6f}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- full_policy clean >= actor_game clean: `{float(full_row['last5_clean_mean']) >= float(actor_row['last5_clean_mean'])}`",
            f"- full_policy force-adv >= actor_game force-adv: `{float(full_row['last5_adversarial_mean']) >= float(actor_row['last5_adversarial_mean'])}`",
        ]
    )
    (output_root / "scope_ablation_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
