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
    label: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 10B targeted proposed-QP tuning")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--stage9-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--scope", type=str, default="full_policy", choices=["full_policy"])
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


def make_candidates(stage5_root: pathlib.Path) -> List[Candidate]:
    configs = load_stage5_configs(stage5_root)
    egm_cfg = configs["egm"]
    lr = float(egm_cfg["lr"])
    max_grad_norm = float(egm_cfg["max_grad_norm"])
    vf_coef = float(egm_cfg["vf_coef"])

    return [
        Candidate(
            "noG_global_base",
            "proposed_noG",
            lr,
            max_grad_norm,
            vf_coef,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "global",
                "qp_g_alpha": 0.3,
                "objective": "loss_decrease",
            },
        ),
        Candidate(
            "noG_global_beta3_normobj",
            "proposed_noG",
            lr,
            max_grad_norm,
            vf_coef,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "global",
                "qp_g_alpha": 0.3,
                "beta_max": 3.0,
                "objective": "normalized_loss_decrease",
            },
        ),
        Candidate(
            "qp_global_base",
            "proposed_qp",
            lr,
            max_grad_norm,
            vf_coef,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "global",
                "qp_g_alpha": 0.3,
                "objective": "loss_decrease",
            },
        ),
        Candidate(
            "qp_global_beta3_gamma3",
            "proposed_qp",
            lr,
            max_grad_norm,
            vf_coef,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "global",
                "qp_g_alpha": 0.3,
                "beta_max": 3.0,
                "gamma_max": 3.0,
                "gamma_scale": 3.0,
                "objective": "loss_decrease",
            },
        ),
        Candidate(
            "qp_global_beta1_gamma10_normobj_cap3",
            "proposed_qp",
            lr,
            max_grad_norm,
            vf_coef,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "global",
                "qp_g_alpha": 1.0,
                "beta_max": 1.0,
                "gamma_max": 10.0,
                "gamma_scale": 3.0,
                "max_update_norm": 3.0,
                "objective": "normalized_loss_decrease",
            },
        ),
        Candidate(
            "qp_block_beta3_gamma3",
            "proposed_qp",
            lr,
            max_grad_norm,
            vf_coef,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "block",
                "qp_g_alpha": 0.3,
                "beta_max": 3.0,
                "gamma_max": 3.0,
                "gamma_scale": 3.0,
                "objective": "loss_decrease",
            },
        ),
        Candidate(
            "qp_block_beta1_gamma10_normobj_cap3",
            "proposed_qp",
            lr,
            max_grad_norm,
            vf_coef,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "block",
                "qp_g_alpha": 1.0,
                "beta_max": 1.0,
                "gamma_max": 10.0,
                "gamma_scale": 3.0,
                "max_update_norm": 3.0,
                "objective": "normalized_loss_decrease",
            },
        ),
    ]


def render_kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens = []
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


def ensure_run(candidate: Candidate, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
    run_root = runs_dir / candidate.label
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
            [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.label],
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
    tokens = render_kwargs_tokens(candidate.optimizer_kwargs)
    if tokens:
        command.extend(["--protagonist-optimizer-kwargs", *tokens])
        command.extend(["--adversary-optimizer-kwargs", *tokens])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
    latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.label],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return latest_run_dir


def collect_metrics(candidate: Candidate, analysis_dir: pathlib.Path, run_dir: pathlib.Path) -> Dict[str, object]:
    summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    pro_metrics = pd.read_csv(run_dir / "analysis" / "protagonist_training_metrics.csv")
    adv_metrics = pd.read_csv(run_dir / "analysis" / "adversary_training_metrics.csv")
    qp_frames = []
    for role_name in ["protagonist", "adversary"]:
        qp_path = run_dir / f"{role_name}_proposed_qp_diagnostics.csv"
        if qp_path.exists():
            frame = pd.read_csv(qp_path)
            frame["role_name"] = role_name
            qp_frames.append(frame)
    qp_df = pd.concat(qp_frames, ignore_index=True) if qp_frames else pd.DataFrame()

    row = {
        "label": candidate.label,
        "method": candidate.optimizer,
        "qp_normalization": candidate.optimizer_kwargs.get("qp_normalization", ""),
        "qp_g_alpha": candidate.optimizer_kwargs.get("qp_g_alpha", np.nan),
        "beta_max": candidate.optimizer_kwargs.get("beta_max", np.inf),
        "gamma_max": candidate.optimizer_kwargs.get("gamma_max", np.inf),
        "gamma_scale": candidate.optimizer_kwargs.get("gamma_scale", 1.0),
        "max_update_norm": candidate.optimizer_kwargs.get("max_update_norm", np.inf),
        "objective": candidate.optimizer_kwargs.get("objective", "loss_decrease"),
        "last5_clean_mean": summary.get("last5_clean_mean", np.nan),
        "last5_control_adv_mean": summary.get("last5_adversarial_mean", np.nan),
        "auc_clean": summary.get("auc_clean", np.nan),
        "auc_adv": summary.get("auc_adversarial", np.nan),
        "crash_flag": summary.get("crash_flag", 1),
        "nan_flag": summary.get("nan_flag", 1),
        "pro_actor_update_norm_mean": float(pro_metrics["actor_update_norm"].mean()),
        "pro_critic_update_norm_mean": float(pro_metrics["critic_update_norm"].mean()),
        "pro_gamma_active_frac_mean": float(pro_metrics.get("gamma_active_frac", pd.Series([0.0])).mean()),
        "pro_G_contribution_norm_mean": float(pro_metrics.get("G_contribution_norm", pd.Series([0.0])).mean()),
        "pro_zero_update_frac": float(pro_metrics.get("zero_update_flag", pd.Series([0.0])).mean()),
        "pro_update_norm_mean": float((pro_metrics["actor_update_norm"] + pro_metrics["critic_update_norm"]).mean()),
        "adv_actor_update_norm_mean": float(adv_metrics["actor_update_norm"].mean()),
        "adv_critic_update_norm_mean": float(adv_metrics["critic_update_norm"].mean()),
        "adv_gamma_active_frac_mean": float(adv_metrics.get("gamma_active_frac", pd.Series([0.0])).mean()),
        "adv_G_contribution_norm_mean": float(adv_metrics.get("G_contribution_norm", pd.Series([0.0])).mean()),
        "adv_zero_update_frac": float(adv_metrics.get("zero_update_flag", pd.Series([0.0])).mean()),
        "adv_update_norm_mean": float((adv_metrics["actor_update_norm"] + adv_metrics["critic_update_norm"]).mean()),
    }
    if not qp_df.empty:
        row.update(
            {
                "same_minibatch_actual_loss_change_mean": float((-qp_df["same_minibatch_total_loss_change"]).mean()),
                "qp_gamma_active_frac_mean": float(qp_df["gamma_active_frac"].mean()),
                "qp_G_contribution_norm_mean": float(qp_df["G_contribution_norm"].mean()),
                "qp_zero_update_frac": float(qp_df["zero_update_flag"].mean()),
                "qp_update_norm_mean": float(qp_df["update_norm_post_cap"].mean()),
            }
        )
    else:
        row.update(
            {
                "same_minibatch_actual_loss_change_mean": np.nan,
                "qp_gamma_active_frac_mean": 0.0,
                "qp_G_contribution_norm_mean": 0.0,
                "qp_zero_update_frac": 0.0,
                "qp_update_norm_mean": np.nan,
            }
        )
    return row


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_root / "stage10b_runs_seed0"

    candidates = make_candidates(pathlib.Path(args.stage5_root))
    rows = []
    for candidate in candidates:
        latest_run_dir = ensure_run(candidate, args, runs_dir)
        analysis_dir = runs_dir / candidate.label / "analysis"
        run_dir = latest_run_dir
        rows.append(collect_metrics(candidate, analysis_dir, run_dir))

    df = pd.DataFrame(rows)
    summary_path = output_root / "stage10b_qp_tuning_summary.csv"
    df.to_csv(summary_path, index=False)

    stage9_df = pd.read_csv(pathlib.Path(args.stage9_root) / "control_training_summary.csv")
    egm_row = stage9_df[stage9_df["method"] == "egm"].iloc[0]

    viable = df[
        (df["nan_flag"] == 0)
        & (df["crash_flag"] == 0)
        & (df["qp_gamma_active_frac_mean"] > 0.0)
        & (df["qp_zero_update_frac"] < 0.5)
        & (df["qp_G_contribution_norm_mean"] > 1e-6)
    ].copy()
    if not viable.empty:
        viable["selection_score"] = (
            viable["last5_control_adv_mean"].fillna(-1e9)
            + 0.25 * viable["last5_clean_mean"].fillna(-1e9)
            + 50.0 * viable["qp_gamma_active_frac_mean"].fillna(0.0)
            - 20.0 * viable["qp_zero_update_frac"].fillna(0.0)
        )
        viable = viable.sort_values("selection_score", ascending=False)
        best_label = str(viable.iloc[0]["label"])
    else:
        best_label = ""

    report_lines = [
        "# Stage 10B targeted proposed-QP tuning",
        "",
        f"- train protocol: `proper control-RARL`",
        f"- scope: `{args.scope}`",
        f"- seed: `{args.seed}`",
        f"- frozen EGM clean last5 mean: `{float(egm_row['last5_clean_mean']):.6f}`",
        f"- frozen EGM control-adv last5 mean: `{float(egm_row['last5_adversarial_mean']):.6f}`",
        f"- selected best label: `{best_label}`",
        "",
        "## Candidates",
        "",
    ]
    for _, row in df.sort_values("last5_control_adv_mean", ascending=False).iterrows():
        report_lines.append(
            f"- `{row['label']}`: clean=`{row['last5_clean_mean']:.6f}`, adv=`{row['last5_control_adv_mean']:.6f}`, "
            f"gamma_active=`{row['qp_gamma_active_frac_mean']:.6f}`, G_contrib=`{row['qp_G_contribution_norm_mean']:.6f}`, "
            f"zero_update=`{row['qp_zero_update_frac']:.6f}`, loss_dec=`{row['same_minibatch_actual_loss_change_mean']:.6f}`"
        )
    (output_root / "stage10b_qp_tuning_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    with open(output_root / "stage10b_selected_qp_config.json", "w", encoding="utf-8") as handle:
        json.dump({"best_label": best_label}, handle, indent=2)

    frontier = df.sort_values("last5_control_adv_mean", ascending=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(frontier["last5_clean_mean"], frontier["last5_control_adv_mean"], c=frontier["qp_gamma_active_frac_mean"], cmap="viridis", s=90)
    for _, row in frontier.iterrows():
        plt.text(row["last5_clean_mean"], row["last5_control_adv_mean"], row["label"], fontsize=8)
    plt.xlabel("last5 clean mean")
    plt.ylabel("last5 control-adv mean")
    plt.title("Stage 10B QP tuning frontier")
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10b_qp_tuning_frontier.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(frontier))
    plt.bar(x - 0.15, frontier["qp_gamma_active_frac_mean"], width=0.3, label="gamma_active_frac")
    plt.bar(x + 0.15, frontier["qp_G_contribution_norm_mean"], width=0.3, label="G_contribution_norm")
    plt.xticks(x, frontier["label"], rotation=25, ha="right")
    plt.title("Stage 10B beta/gamma diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10b_beta_gamma_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
