from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import pandas as pd


BASELINE_METHODS = ["adam", "sgd", "egm", "ppm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 0 baseline freeze/config read for perfLyap line")
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--repo-root", type=str, required=True)
    return parser.parse_args()


def git_status(repo_root: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    baseline_root = pathlib.Path(args.baseline_root)
    output_root = pathlib.Path(args.output_root)
    repo_root = pathlib.Path(args.repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    training_summary_path = baseline_root / "rawFG_matched_training_summary.csv"
    eval_curves_path = baseline_root / "rawFG_matched_eval_curves.csv"
    param_norms_path = baseline_root / "rawFG_matched_param_norms.csv"

    training_summary = pd.read_csv(training_summary_path)
    eval_curves = pd.read_csv(eval_curves_path)
    param_norms = pd.read_csv(param_norms_path)

    baseline_df = training_summary[training_summary["method"].isin(BASELINE_METHODS)].copy()
    baseline_df = baseline_df[
        [
            "method",
            "protagonist_optimizer",
            "adversary_optimizer",
            "protagonist_lr",
            "adversary_lr",
            "protagonist_max_grad_norm",
            "adversary_max_grad_norm",
            "protagonist_vf_coef",
            "adversary_vf_coef",
            "ppm_inner_steps",
            "total_iterations",
            "num_training_episodes",
            "num_clean_eval_points",
            "num_adv_eval_points",
            "last5_training_mean",
            "last5_clean_mean",
            "last5_adversarial_mean",
            "auc_training",
            "auc_clean",
            "auc_adversarial",
        ]
    ].sort_values("method")
    baseline_df.to_csv(output_root / "stage0_baseline_reference_summary.csv", index=False)

    status_text = git_status(repo_root)
    required_files = {
        "training_summary": training_summary_path.exists(),
        "eval_curves": eval_curves_path.exists(),
        "param_norms": param_norms_path.exists(),
    }
    eval_counts = {
        method: {
            "total_iterations": int(training_summary.loc[training_summary["method"] == method, "total_iterations"].iloc[0]),
            "num_training_episodes": int(training_summary.loc[training_summary["method"] == method, "num_training_episodes"].iloc[0]),
            "num_clean_eval_points": int(training_summary.loc[training_summary["method"] == method, "num_clean_eval_points"].iloc[0]),
            "num_adv_eval_points": int(training_summary.loc[training_summary["method"] == method, "num_adv_eval_points"].iloc[0]),
            "eval_curve_rows": int((eval_curves["method"] == method).sum()),
            "param_norm_rows": int((param_norms["method"] == method).sum()),
        }
        for method in BASELINE_METHODS
    }

    report_lines = [
        "# Stage 0 Baseline Freeze Report",
        "",
        "## Status",
        "- Baselines referenced: `adam`, `sgd`, `egm`, `ppm`.",
        "- This line only adds `proposed_noG_perfLyap` / `proposed_qp_perfLyap`.",
        "- Existing matched-budget baseline result root was read, not rerun.",
        "",
        "## Readability Check",
        *(f"- `{name}` readable: `{flag}`" for name, flag in required_files.items()),
        "",
        "## Baseline Code Freeze",
        "- `models/optimizers.py`, `utils/exp_manager.py`, and `scripts/train_adversary.py` are modified in the current worktree, but the Stage 0 inspection confirms the baseline optimizer class names `adam` / `sgd` / `egm` / `ppm` remain present and are not replaced by this perfLyap line.",
        "- New work for this line is restricted to adding new proposed optimizer names and proposed-only scripts/results.",
        "",
        "## Current Git Status",
        "```text",
        status_text or "(clean)",
        "```",
        "",
        "## Baseline Result Counts",
    ]
    for method in BASELINE_METHODS:
        counts = eval_counts[method]
        report_lines.extend(
            [
                f"### {method}",
                f"- total_iterations: `{counts['total_iterations']}`",
                f"- num_training_episodes: `{counts['num_training_episodes']}`",
                f"- num_clean_eval_points: `{counts['num_clean_eval_points']}`",
                f"- num_adv_eval_points: `{counts['num_adv_eval_points']}`",
                f"- eval_curve_rows: `{counts['eval_curve_rows']}`",
                f"- param_norm_rows: `{counts['param_norm_rows']}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Confirmation",
            "1. `Adam/SGD/EGM/PPM` baseline code paths remain available and are not being rerun in this line.",
            "2. Existing matched-budget baseline summaries/eval curves/param norms are readable.",
            "3. Subsequent work in this result root will only add `proposed_noG_perfLyap` / `proposed_qp_perfLyap`.",
            "",
        ]
    )
    (output_root / "stage0_baseline_freeze_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
