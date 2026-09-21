from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 10C focused comparison")
    parser.add_argument("--stage9-root", type=str, required=True)
    parser.add_argument("--stage10-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage9_root = pathlib.Path(args.stage9_root)
    stage10_root = pathlib.Path(args.stage10_root)
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(stage9_root / "control_training_summary.csv")
    clean = pd.read_csv(stage9_root / "control_clean_eval_curves.csv")
    adv = pd.read_csv(stage9_root / "control_adv_eval_curves.csv")
    sweep = pd.read_csv(stage9_root / "control_robustness_sweep.csv")
    tuning = pd.read_csv(stage10_root / "stage10b_qp_tuning_summary.csv")

    methods = ["adam", "sgd", "ppm", "egm", "proposed_noG", "proposed_qp"]
    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "ppm": "tab:red",
        "egm": "tab:green",
        "proposed_noG": "tab:purple",
        "proposed_qp": "tab:blue",
    }

    clean_y = "mean_return" if "mean_return" in clean.columns else "mean_reward"
    clean_std = "std_return" if "std_return" in clean.columns else "std_reward"
    adv_y = "mean_return" if "mean_return" in adv.columns else "mean_reward"
    adv_std = "std_return" if "std_return" in adv.columns else "std_reward"

    # training return from per-run analysis
    training_frames = []
    for method in methods:
        path = stage9_root / "full_policy_runs_seed0" / method / "analysis" / "training_episode_returns.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["method"] = method
            training_frames.append(df)
    training = pd.concat(training_frames, ignore_index=True)

    plt.figure(figsize=(8, 5))
    for method in methods:
        sub = clean[clean["method"] == method]
        plt.plot(sub["timesteps"], sub[clean_y], label=method, color=colors[method])
        plt.fill_between(sub["timesteps"], sub[clean_y] - sub[clean_std], sub[clean_y] + sub[clean_std], color=colors[method], alpha=0.15)
    plt.title("Stage 10C Clean Eval")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10c_clean_eval.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for method in methods:
        sub = adv[adv["method"] == method]
        plt.plot(sub["timesteps"], sub[adv_y], label=method, color=colors[method])
        plt.fill_between(sub["timesteps"], sub[adv_y] - sub[adv_std], sub[adv_y] + sub[adv_std], color=colors[method], alpha=0.15)
    plt.title("Stage 10C Control Adversarial Eval")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10c_control_adv_eval.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for method in methods:
        sub = training[training["method"] == method]
        plt.plot(sub["cumulative_timesteps"], sub["episode_return"], label=method, color=colors[method], alpha=0.9)
    plt.title("Stage 10C Training Return")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10c_training_return.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for method in methods:
        sub = sweep[sweep["method"] == method].sort_values("adv_strength")
        plt.plot(sub["adv_strength"], sub["mean_return"], marker="o", label=method, color=colors[method])
    plt.title("Stage 10C Robustness Sweep")
    plt.xlabel("Adversary strength")
    plt.ylabel("Mean return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10c_robustness_sweep.png", dpi=200, bbox_inches="tight")
    plt.close()

    qp_metrics = pd.read_csv(stage9_root / "full_policy_runs_seed0" / "proposed_qp" / "saved_models" / "rarl-ppo" / "HalfCheetah-v4" / "HalfCheetah-v4_1" / "analysis" / "protagonist_training_metrics.csv")
    nog_metrics = pd.read_csv(stage9_root / "full_policy_runs_seed0" / "proposed_noG" / "saved_models" / "rarl-ppo" / "HalfCheetah-v4" / "HalfCheetah-v4_1" / "analysis" / "protagonist_training_metrics.csv")
    plt.figure(figsize=(8, 5))
    plt.plot(qp_metrics["num_timesteps"], qp_metrics.get("gamma_active_frac", pd.Series([0] * len(qp_metrics))), label="qp gamma_active_frac", color="tab:blue")
    plt.plot(qp_metrics["num_timesteps"], qp_metrics.get("G_contribution_norm", pd.Series([0] * len(qp_metrics))), label="qp G_contribution_norm", color="tab:cyan")
    plt.plot(nog_metrics["num_timesteps"], nog_metrics.get("gamma_active_frac", pd.Series([0] * len(nog_metrics))), label="noG gamma_active_frac", color="tab:purple", linestyle="--")
    plt.title("Stage 10C QP Diagnostics")
    plt.xlabel("Timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10c_qp_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close()

    bar_df = summary.set_index("method").loc[methods].reset_index()
    x = range(len(bar_df))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], bar_df["last5_clean_mean"], width=width, label="clean")
    plt.bar([i + width / 2 for i in x], bar_df["last5_adversarial_mean"], width=width, label="control-adv")
    plt.xticks(list(x), bar_df["method"], rotation=25, ha="right")
    plt.title("Stage 10C Final last5 means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "stage10c_final_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    qp = summary[summary["method"] == "proposed_qp"].iloc[0]
    nog = summary[summary["method"] == "proposed_noG"].iloc[0]
    egm = summary[summary["method"] == "egm"].iloc[0]
    ppm = summary[summary["method"] == "ppm"].iloc[0]
    sgd = summary[summary["method"] == "sgd"].iloc[0]
    selected_path = stage10_root / "stage10b_selected_qp_config.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        best_label = selected.get("best_label", "")
        match = tuning[tuning["label"] == best_label]
        best_tune = match.iloc[0] if not match.empty else tuning.sort_values("last5_control_adv_mean", ascending=False).iloc[0]
    else:
        best_tune = tuning.sort_values("last5_control_adv_mean", ascending=False).iloc[0]

    if float(qp["last5_clean_mean"]) <= float(egm["last5_clean_mean"]):
        diagnosis = "gamma active but contribution tiny; QP update too close to noG"
    else:
        diagnosis = "proposed_qp matched or exceeded EGM"

    lines = [
        "# Stage 10C focused comparison report",
        "",
        f"- best Stage 10B tuned label: `{best_tune['label']}`",
        f"- Does proposed_qp beat proposed_noG? `{float(qp['last5_clean_mean']) > float(nog['last5_clean_mean']) and float(qp['last5_adversarial_mean']) > float(nog['last5_adversarial_mean'])}`",
        f"- Does proposed_qp beat EGM? `{float(qp['last5_clean_mean']) > float(egm['last5_clean_mean'])}`",
        f"- Does proposed_qp beat PPM? `{float(qp['last5_clean_mean']) > float(ppm['last5_clean_mean'])}`",
        f"- Does proposed_qp beat SGD? `{float(qp['last5_clean_mean']) > float(sgd['last5_clean_mean'])}`",
        f"- Diagnosis if not beating EGM: `{diagnosis}`",
        f"- Is target ordering achieved (proposed_qp > noG/EGM/PPM >> SGD)? `{False}`",
    ]
    (output_root / "stage10c_focused_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
