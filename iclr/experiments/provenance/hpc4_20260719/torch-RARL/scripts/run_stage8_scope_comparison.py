from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 8F cross-scope comparison")
    parser.add_argument("--output-root", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    full_summary = pd.read_csv(output_root / "full_policy_all_methods_summary.csv")
    actor_summary = pd.read_csv(output_root / "actor_game_all_methods_summary.csv")
    full_diag = pd.read_csv(output_root / "full_policy_qp_diagnostics.csv")
    actor_diag = pd.read_csv(output_root / "actor_game_qp_diagnostics.csv")
    full_updates = pd.read_csv(output_root / "full_policy_update_diagnostics.csv")
    actor_updates = pd.read_csv(output_root / "actor_game_update_diagnostics.csv")

    full_summary["scope"] = "full_policy"
    actor_summary["scope"] = "actor_game"
    summary_df = pd.concat([full_summary, actor_summary], ignore_index=True)
    summary_df.to_csv(output_root / "scope_comparison_summary.csv", index=False)

    full_qp = full_diag[full_diag["method"] == "proposed_qp"].copy()
    actor_qp = actor_diag[actor_diag["method"] == "proposed_qp"].copy()
    full_qp["scope"] = "full_policy"
    actor_qp["scope"] = "actor_game"
    qp_df = pd.concat([full_qp, actor_qp], ignore_index=True)

    full_updates["scope"] = "full_policy"
    actor_updates["scope"] = "actor_game"
    updates_df = pd.concat([full_updates, actor_updates], ignore_index=True)

    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    methods = summary_df["method"].unique().tolist()
    x = np.arange(len(methods))
    width = 0.35

    for idx, (metric, title) in enumerate(
        [
            ("last5_clean_mean", "final clean eval"),
            ("last5_adversarial_mean", "final force adv eval"),
            ("last5_control_proxy_mean", "final control-proxy eval"),
        ]
    ):
        ax = axes[idx // 2, idx % 2]
        full_vals = full_summary.set_index("method").reindex(methods)[metric].to_numpy()
        actor_vals = actor_summary.set_index("method").reindex(methods)[metric].to_numpy()
        ax.bar(x - width / 2, full_vals, width=width, label="full_policy")
        ax.bar(x + width / 2, actor_vals, width=width, label="actor_game")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    beta_means = qp_df.groupby("scope")["beta"].mean()
    gamma_means = qp_df.groupby("scope")["gamma"].mean()
    axes[1, 1].bar(["full_beta", "actor_beta", "full_gamma", "actor_gamma"], [beta_means.get("full_policy", np.nan), beta_means.get("actor_game", np.nan), gamma_means.get("full_policy", np.nan), gamma_means.get("actor_game", np.nan)])
    axes[1, 1].set_title("proposed_qp beta/gamma statistics by scope")
    axes[1, 1].grid(alpha=0.3)

    gamma_active = qp_df.groupby("scope")["gamma_active_frac"].mean()
    zero_update = qp_df.groupby("scope")["zero_update_flag"].mean()
    axes[2, 0].bar(["full_gamma_active", "actor_gamma_active", "full_zero", "actor_zero"], [gamma_active.get("full_policy", np.nan), gamma_active.get("actor_game", np.nan), zero_update.get("full_policy", np.nan), zero_update.get("actor_game", np.nan)])
    axes[2, 0].set_title("gamma_active_frac / zero_update_frac by scope")
    axes[2, 0].grid(alpha=0.3)

    block_update = updates_df.groupby(["scope", "method"])[["actor_update_norm", "critic_update_norm"]].mean().reset_index()
    for scope, color in [("full_policy", "tab:blue"), ("actor_game", "tab:orange")]:
        group = block_update[block_update["scope"] == scope]
        axes[2, 1].plot(group["method"], group["actor_update_norm"], marker="o", label=f"{scope}_actor", color=color)
        axes[2, 1].plot(group["method"], group["critic_update_norm"], marker="s", linestyle="--", label=f"{scope}_critic", color=color)
    axes[2, 1].set_title("update norm by block and scope")
    axes[2, 1].tick_params(axis="x", rotation=35)
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].legend(fontsize=8)

    rank_df = summary_df[["scope", "method", "last5_clean_mean", "last5_adversarial_mean", "last5_control_proxy_mean"]].copy()
    rank_df["rank_clean"] = rank_df.groupby("scope")["last5_clean_mean"].rank(ascending=False, method="min")
    rank_df["rank_force"] = rank_df.groupby("scope")["last5_adversarial_mean"].rank(ascending=False, method="min")
    rank_df["rank_control"] = rank_df.groupby("scope")["last5_control_proxy_mean"].rank(ascending=False, method="min")
    rank_df.to_csv(output_root / "scope_ranking_table.csv", index=False)
    axes[3, 0].axis("off")
    table = axes[3, 0].table(
        cellText=rank_df[["scope", "method", "rank_clean", "rank_force", "rank_control"]].values,
        colLabels=["scope", "method", "rank_clean", "rank_force", "rank_control"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    axes[3, 0].set_title("ranking table")

    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.02,
        0.98,
        "\n".join(
            [
                "Target ordering:",
                "proposed_qp >> ppm/egm/noG > sgd",
                "",
                f"full_policy proposed_qp > noG: {bool(float(full_summary.set_index('method').loc['proposed_qp', 'last5_clean_mean']) > float(full_summary.set_index('method').loc['proposed_noG', 'last5_clean_mean']))}",
                f"actor_game proposed_qp > noG: {bool(float(actor_summary.set_index('method').loc['proposed_qp', 'last5_clean_mean']) > float(actor_summary.set_index('method').loc['proposed_noG', 'last5_clean_mean']))}",
            ]
        ),
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(plots_dir / "scope_comparison_big.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    full_qp_row = full_summary.set_index("method").loc["proposed_qp"]
    actor_qp_row = actor_summary.set_index("method").loc["proposed_qp"]
    full_nog_row = full_summary.set_index("method").loc["proposed_noG"]
    actor_nog_row = actor_summary.set_index("method").loc["proposed_noG"]
    full_egm_row = full_summary.set_index("method").loc["egm"]
    actor_egm_row = actor_summary.set_index("method").loc["egm"]
    full_ppm_row = full_summary.set_index("method").loc["ppm"]
    actor_ppm_row = actor_summary.set_index("method").loc["ppm"]
    full_sgd_row = full_summary.set_index("method").loc["sgd"]
    actor_sgd_row = actor_summary.set_index("method").loc["sgd"]

    better_scope = "actor_game" if float(actor_qp_row["last5_clean_mean"]) >= float(full_qp_row["last5_clean_mean"]) else "full_policy"
    report_lines = [
        "# Stage 8 Scope Comparison",
        "",
        f"- better scope for proposed_qp by clean last5 mean: `{better_scope}`",
        f"- proposed_qp beats proposed_noG in full_policy: `{float(full_qp_row['last5_clean_mean']) > float(full_nog_row['last5_clean_mean'])}`",
        f"- proposed_qp beats proposed_noG in actor_game: `{float(actor_qp_row['last5_clean_mean']) > float(actor_nog_row['last5_clean_mean'])}`",
        f"- proposed_qp beats EGM in full_policy: `{float(full_qp_row['last5_clean_mean']) > float(full_egm_row['last5_clean_mean'])}`",
        f"- proposed_qp beats PPM in full_policy: `{float(full_qp_row['last5_clean_mean']) > float(full_ppm_row['last5_clean_mean'])}`",
        f"- proposed_qp beats EGM in actor_game: `{float(actor_qp_row['last5_clean_mean']) > float(actor_egm_row['last5_clean_mean'])}`",
        f"- proposed_qp beats PPM in actor_game: `{float(actor_qp_row['last5_clean_mean']) > float(actor_ppm_row['last5_clean_mean'])}`",
        f"- proposed_qp beats SGD in full_policy: `{float(full_qp_row['last5_clean_mean']) > float(full_sgd_row['last5_clean_mean'])}`",
        f"- proposed_qp beats SGD in actor_game: `{float(actor_qp_row['last5_clean_mean']) > float(actor_sgd_row['last5_clean_mean'])}`",
        "",
        "## Recommendation",
        "",
        "- Stage 8 is still a force-trained optimizer/scope benchmark, not the final adversary-protocol result.",
        f"- Next formal experiment should use: `{'proper control-RARL retraining' if True else 'force'}`",
    ]
    (output_root / "scope_comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
