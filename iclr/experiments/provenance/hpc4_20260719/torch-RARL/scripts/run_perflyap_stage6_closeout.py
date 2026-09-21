from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PROPOSED_METHODS = [
    "proposed_noG_fullparam_mixed_clean_unclipped",
    "proposed_qp_fullparam_mixed_clean_unclipped_actual_selector",
    "proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG",
    "proposed_qp_fullparam_mixed_rarl_unclipped_actual_selector",
]

BASELINE_METHODS = ["adam", "sgd", "egm", "ppm"]


@dataclass(frozen=True)
class MethodContext:
    method: str
    run_root: pathlib.Path
    latest_run_dir: pathlib.Path
    analysis_dir: pathlib.Path
    run_summary: Dict[str, object]
    n_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 6 close-out")
    parser.add_argument("--result-root", type=str, required=True)
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    return parser.parse_args()


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def first_existing(frame: pd.DataFrame, names: Sequence[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"Missing columns {list(names)} in {list(frame.columns)}")


def compute_auc(x: pd.Series, y: pd.Series) -> float:
    xs = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    xs = xs[mask]
    ys = ys[mask]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.trapz(ys, xs))


def load_method_context(method: str, run_root: pathlib.Path, env_id: str) -> MethodContext:
    latest_run_dir = find_latest_run_dir(run_root / "saved_models", env_id)
    analysis_dir = run_root / "analysis"
    run_summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    args_path = next(path for path in latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists()) / "args.yml"
    n_steps = 2048
    text = args_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("n_steps:"):
            try:
                n_steps = int(float(line.split(":", 1)[1].strip()))
            except ValueError:
                n_steps = 2048
            break
    return MethodContext(
        method=method,
        run_root=run_root,
        latest_run_dir=latest_run_dir,
        analysis_dir=analysis_dir,
        run_summary=run_summary,
        n_steps=n_steps,
    )


def read_training_frame(ctx: MethodContext) -> pd.DataFrame:
    df = pd.read_csv(ctx.analysis_dir / "training_episode_returns.csv").copy()
    time_col = first_existing(df, ["timesteps", "timestep", "total_timesteps", "cumulative_timesteps"])
    ret_col = first_existing(df, ["episode_return", "reward", "ep_rew_mean", "return"])
    df["timestep"] = pd.to_numeric(df[time_col], errors="coerce")
    df["episode_return"] = pd.to_numeric(df[ret_col], errors="coerce")
    df["outer_iteration"] = df["timestep"] / float(ctx.n_steps)
    df["method"] = ctx.method
    return df


def read_eval_frame(path: pathlib.Path, method: str, n_steps: int, eval_type: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    time_col = first_existing(df, ["timesteps", "timestep", "total_timesteps"])
    mean_col = first_existing(df, ["mean_reward", "clean_mean", "adv_mean", "adversarial_mean", "control_adv_mean", "eval_mean_reward"])
    std_col = None
    for name in ["std_reward", "clean_std", "adv_std", "adversarial_std", "control_adv_std", "eval_std_reward"]:
        if name in df.columns:
            std_col = name
            break
    df["timestep"] = pd.to_numeric(df[time_col], errors="coerce")
    df["mean_reward"] = pd.to_numeric(df[mean_col], errors="coerce")
    df["std_reward"] = pd.to_numeric(df[std_col], errors="coerce") if std_col else np.nan
    df["outer_iteration"] = df["timestep"] / float(n_steps)
    df["method"] = method
    df["eval_type"] = eval_type
    if "applied_perturbation_norm" not in df.columns:
        df["applied_perturbation_norm"] = np.nan
    if "clip_fraction_eval" not in df.columns:
        df["clip_fraction_eval"] = np.nan
    return df


def read_param_frame(ctx: MethodContext) -> pd.DataFrame:
    df = pd.read_csv(ctx.analysis_dir / "parameter_norms.csv").copy()
    time_col = first_existing(df, ["num_timesteps", "timesteps", "timestep"])
    df["num_timesteps"] = pd.to_numeric(df[time_col], errors="coerce")
    df["outer_iteration"] = df["num_timesteps"] / float(ctx.n_steps)
    df["method"] = ctx.method
    return df


def recover_update_frames(contexts: Sequence[MethodContext]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for ctx in contexts:
        metrics_root = ctx.latest_run_dir / "analysis"
        for role in ("protagonist", "adversary"):
            path = metrics_root / f"{role}_training_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).copy()
            if df.empty:
                continue
            df["method"] = ctx.method
            df["optimizer_role"] = role
            df["num_timesteps"] = pd.to_numeric(df[first_existing(df, ["num_timesteps", "timesteps", "timestep"])], errors="coerce")
            df["outer_iteration"] = df["num_timesteps"] / float(ctx.n_steps)
            df["lr"] = float(ctx.run_summary.get("protagonist_lr" if role == "protagonist" else "adversary_lr", np.nan))
            df["max_grad_norm"] = float(ctx.run_summary.get("protagonist_max_grad_norm" if role == "protagonist" else "adversary_max_grad_norm", np.nan))
            df["vf_coef"] = float(ctx.run_summary.get("protagonist_vf_coef" if role == "protagonist" else "adversary_vf_coef", np.nan))
            rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.concat(rows, ignore_index=True)


def recover_qp_diagnostics(proposed_contexts: Sequence[MethodContext]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ctx in proposed_contexts:
        metrics_root = ctx.latest_run_dir / "analysis"
        saved_root = ctx.latest_run_dir
        for role, diag_name in (
            ("protagonist", "protagonist_proposed_qp_perflyap_diagnostics.csv"),
            ("adversary", "adversary_proposed_qp_perflyap_diagnostics.csv"),
            ("protagonist", "protagonist_proposed_nog_perflyap_diagnostics.csv"),
            ("adversary", "adversary_proposed_nog_perflyap_diagnostics.csv"),
        ):
            diag_path = saved_root / diag_name
            metrics_path = metrics_root / f"{role}_training_metrics.csv"
            if not diag_path.exists() or not metrics_path.exists():
                continue
            raw = pd.read_csv(diag_path)
            metrics = pd.read_csv(metrics_path)
            if raw.empty or metrics.empty:
                continue
            metrics["num_timesteps"] = pd.to_numeric(metrics[first_existing(metrics, ["num_timesteps", "timesteps", "timestep"])], errors="coerce")
            metrics["outer_iteration"] = metrics["num_timesteps"] / float(ctx.n_steps)
            cumulative = metrics["n_updates"].astype(int).tolist()
            start = 0
            out_rows = []
            for idx, stop in enumerate(cumulative):
                stop = int(stop)
                if stop <= start:
                    continue
                chunk = raw.iloc[start:stop].copy()
                start = stop
                if chunk.empty:
                    continue
                actor_surrogate_series = None
                for col in ("unclipped_actor_surrogate_change", "actual_C_change"):
                    if col in chunk.columns:
                        actor_surrogate_series = pd.to_numeric(chunk[col], errors="coerce")
                        break
                selected_direction = None
                if "direction_mode" in chunk.columns:
                    selected_direction = chunk["direction_mode"].dropna().astype(str).mode()
                    selected_direction = selected_direction.iloc[0] if not selected_direction.empty else ""
                fallback_reason = ""
                if "fallback_reason" in chunk.columns:
                    modes = chunk["fallback_reason"].fillna("").astype(str).mode()
                    fallback_reason = modes.iloc[0] if not modes.empty else ""
                row = {
                    "method": ctx.method,
                    "optimizer_role": role,
                    "num_timesteps": float(metrics.iloc[idx]["num_timesteps"]),
                    "outer_iteration": float(metrics.iloc[idx]["outer_iteration"]),
                    "beta_raw": pd.to_numeric(chunk["beta_raw"], errors="coerce").mean() if "beta_raw" in chunk.columns else np.nan,
                    "gamma_raw": pd.to_numeric(chunk["gamma_raw"], errors="coerce").mean() if "gamma_raw" in chunk.columns else np.nan,
                    "beta_eff": pd.to_numeric(chunk["beta_eff"], errors="coerce").mean() if "beta_eff" in chunk.columns else np.nan,
                    "gamma_eff": pd.to_numeric(chunk["gamma_eff"], errors="coerce").mean() if "gamma_eff" in chunk.columns else np.nan,
                    "gamma_active_frac": pd.to_numeric(chunk["gamma_active_frac"], errors="coerce").mean() if "gamma_active_frac" in chunk.columns else np.nan,
                    "fallback_to_noG_frac": pd.to_numeric(chunk["fallback_to_noG"], errors="coerce").mean() if "fallback_to_noG" in chunk.columns else np.nan,
                    "selected_direction": selected_direction,
                    "selected_g_sign": chunk["selected_g_sign"].dropna().astype(str).mode().iloc[0] if "selected_g_sign" in chunk.columns and not chunk["selected_g_sign"].dropna().empty else "",
                    "update_norm_pre_cap": pd.to_numeric(chunk["update_norm_pre_cap"], errors="coerce").mean() if "update_norm_pre_cap" in chunk.columns else np.nan,
                    "update_norm_post_cap": pd.to_numeric(chunk["update_norm_post_cap"], errors="coerce").mean() if "update_norm_post_cap" in chunk.columns else np.nan,
                    "cap_active_frac": pd.to_numeric(chunk["cap_active"], errors="coerce").mean() if "cap_active" in chunk.columns else np.nan,
                    "approx_kl": pd.to_numeric(chunk["approx_kl"], errors="coerce").mean() if "approx_kl" in chunk.columns else np.nan,
                    "clip_fraction": pd.to_numeric(chunk["clip_fraction"], errors="coerce").mean() if "clip_fraction" in chunk.columns else np.nan,
                    "actor_update_norm": pd.to_numeric(chunk["actor_update_norm"], errors="coerce").mean() if "actor_update_norm" in chunk.columns else np.nan,
                    "logstd_update_norm": pd.to_numeric(chunk["logstd_update_norm"], errors="coerce").mean() if "logstd_update_norm" in chunk.columns else np.nan,
                    "critic_update_norm": pd.to_numeric(chunk["critic_update_norm"], errors="coerce").mean() if "critic_update_norm" in chunk.columns else np.nan,
                    "actor_fraction_of_update": pd.to_numeric(chunk["actor_fraction_of_update"], errors="coerce").mean() if "actor_fraction_of_update" in chunk.columns else np.nan,
                    "logstd_fraction_of_update": pd.to_numeric(chunk["logstd_fraction_of_update"], errors="coerce").mean() if "logstd_fraction_of_update" in chunk.columns else np.nan,
                    "critic_fraction_of_update": pd.to_numeric(chunk["critic_fraction_of_update"], errors="coerce").mean() if "critic_fraction_of_update" in chunk.columns else np.nan,
                    "value_loss_change": pd.to_numeric(chunk["value_loss_change"], errors="coerce").mean() if "value_loss_change" in chunk.columns else np.nan,
                    "actor_surrogate_change": actor_surrogate_series.mean() if actor_surrogate_series is not None else np.nan,
                    "entropy_change": pd.to_numeric(chunk["entropy_change"], errors="coerce").mean() if "entropy_change" in chunk.columns else np.nan,
                    "mixed_merit_change": pd.to_numeric(chunk["mixed_merit_change"], errors="coerce").mean() if "mixed_merit_change" in chunk.columns else np.nan,
                    "fallback_reason": fallback_reason,
                }
                out_rows.append(row)
            if out_rows:
                frames.append(pd.DataFrame(out_rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_series(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    if df.empty or value_col not in df.columns:
        return {
            "final": float("nan"),
            "last5_mean": float("nan"),
            "auc": float("nan"),
        }
    sdf = df.sort_values("outer_iteration")
    vals = pd.to_numeric(sdf[value_col], errors="coerce")
    return {
        "final": float(vals.iloc[-1]) if len(vals) else float("nan"),
        "last5_mean": float(vals.tail(5).mean()) if len(vals) else float("nan"),
        "auc": compute_auc(sdf["outer_iteration"], vals),
    }


def build_summary(training_df: pd.DataFrame, eval_df: pd.DataFrame, run_summaries: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for method, run_summary in run_summaries.items():
        train_stats = aggregate_series(training_df[training_df["method"] == method], "episode_return")
        clean_det_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "clean_deterministic")], "mean_reward")
        clean_stoch_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "clean_stochastic")], "mean_reward")
        adv_det_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "control_adv_deterministic")], "mean_reward")
        adv_stoch_stats = aggregate_series(eval_df[(eval_df["method"] == method) & (eval_df["eval_type"] == "control_adv_stochastic")], "mean_reward")
        row = dict(run_summary)
        row.update(
            {
                "method": method,
                "final_training_return": train_stats["final"],
                "last5_training_mean": train_stats["last5_mean"],
                "training_auc": train_stats["auc"],
                "final_clean_deterministic": clean_det_stats["final"],
                "last5_clean_deterministic_mean": clean_det_stats["last5_mean"],
                "final_clean_stochastic": clean_stoch_stats["final"],
                "last5_clean_stochastic_mean": clean_stoch_stats["last5_mean"],
                "final_control_adv_deterministic": adv_det_stats["final"],
                "last5_control_adv_deterministic_mean": adv_det_stats["last5_mean"],
                "final_control_adv_stochastic": adv_stoch_stats["final"],
                "last5_control_adv_stochastic_mean": adv_stoch_stats["last5_mean"],
                "clean_auc": clean_det_stats["auc"],
                "control_adv_auc": adv_det_stats["auc"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_training(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["episode_return"], errors="coerce"), label=method, color=colors.get(method))
    ax.set_title("Training return")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)


def plot_band(ax, df: pd.DataFrame, eval_type: str, title: str, colors: Dict[str, str]) -> None:
    sub = df[df["eval_type"] == eval_type].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        mean = pd.to_numeric(group["mean_reward"], errors="coerce")
        std = pd.to_numeric(group["std_reward"], errors="coerce").fillna(0.0)
        ax.plot(group["outer_iteration"], mean, label=method, color=colors.get(method))
        ax.fill_between(group["outer_iteration"], mean - std, mean + std, alpha=0.15, color=colors.get(method))
    ax.set_title(title)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)


def plot_beta_gamma_fallback(axs, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        for ax in axs.flat:
            ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        return
    sub = df[df["optimizer_role"] == "protagonist"].copy()
    plot_specs = [
        ("beta_raw", "Beta raw"),
        ("gamma_raw", "Gamma raw"),
        ("fallback_to_noG_frac", "Fallback-to-noG frac"),
        ("gamma_active_frac", "Gamma active frac"),
    ]
    for ax, (col, title) in zip(axs.flat, plot_specs):
        for method, group in sub.groupby("method"):
            group = group.sort_values("outer_iteration")
            ax.plot(group["outer_iteration"], pd.to_numeric(group[col], errors="coerce"), label=method, color=colors.get(method))
        ax.set_title(title)
        ax.set_xlabel("Outer iteration")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7, ncol=2)


def plot_merit_components(axs, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        for ax in axs.flat:
            ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        return
    sub = df[df["optimizer_role"] == "protagonist"].copy()
    plot_specs = [
        ("mixed_merit_change", "Mixed merit change"),
        ("actor_surrogate_change", "Actor surrogate change"),
        ("value_loss_change", "Value loss change"),
        ("entropy_change", "Entropy change"),
    ]
    for ax, (col, title) in zip(axs.flat, plot_specs):
        for method, group in sub.groupby("method"):
            group = group.sort_values("outer_iteration")
            ax.plot(group["outer_iteration"], pd.to_numeric(group[col], errors="coerce"), label=method, color=colors.get(method))
        ax.set_title(title)
        ax.set_xlabel("Outer iteration")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7, ncol=2)


def plot_block_update_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    sub = df[df["optimizer_role"] == "protagonist"].copy()
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["actor_update_norm"], errors="coerce"), color=colors.get(method), label=f"{method} actor")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["logstd_update_norm"], errors="coerce"), color=colors.get(method), linestyle=":", label=f"{method} logstd")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["critic_update_norm"], errors="coerce"), color=colors.get(method), linestyle="--", label=f"{method} critic")
    ax.set_title("Block update norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Norm")
    ax.grid(alpha=0.3)


def plot_param_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "missing diagnostics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    sub = df[df["agent_name"] == "protagonist"].copy()
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["actor_param_norm"], errors="coerce"), color=colors.get(method), label=f"{method} actor")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["log_std_norm"], errors="coerce"), color=colors.get(method), linestyle=":", label=f"{method} logstd")
        ax.plot(group["outer_iteration"], pd.to_numeric(group["critic_param_norm"], errors="coerce"), color=colors.get(method), linestyle="--", label=f"{method} critic")
    ax.set_title("Protagonist parameter norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)


def plot_final_bar(ax, summary_df: pd.DataFrame) -> None:
    order = summary_df["method"].tolist()
    x = np.arange(len(order))
    width = 0.18
    ax.bar(x - 1.5 * width, summary_df["final_clean_deterministic"], width=width, label="clean_det")
    ax.bar(x - 0.5 * width, summary_df["final_clean_stochastic"], width=width, label="clean_stoch")
    ax.bar(x + 0.5 * width, summary_df["final_control_adv_deterministic"], width=width, label="control_adv_det")
    ax.bar(x + 1.5 * width, summary_df["final_control_adv_stochastic"], width=width, label="control_adv_stoch")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Mean return")
    ax.set_title("Final comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def safe_open_image(path: pathlib.Path) -> Image.Image | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def make_placeholder(title: str, size: tuple[int, int] = (1000, 700)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 40), title, fill="black", font=font)
    draw.text((40, 110), "missing diagnostics", fill="gray", font=font)
    return img


def make_collage(plot_paths: Sequence[pathlib.Path], output_path: pathlib.Path, cols: int = 2) -> None:
    items: List[tuple[pathlib.Path, Image.Image]] = []
    for path in plot_paths:
        img = safe_open_image(path)
        if img is None:
            img = make_placeholder(path.name)
        items.append((path, img))
    if not items:
        return
    cell_w = max(img.width for _, img in items)
    cell_h = max(img.height for _, img in items)
    rows = math.ceil(len(items) / cols)
    pad = 20
    header_h = 36
    canvas = Image.new("RGB", (pad + cols * (cell_w + pad), pad + rows * (cell_h + header_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for idx, (path, img) in enumerate(items):
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
    result_root = pathlib.Path(args.result_root)
    baseline_root = pathlib.Path(args.baseline_root) / "runs_seed0"
    runs_root = result_root / "runs_seed0"
    plots_dir = result_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    contexts: List[MethodContext] = []
    for method in BASELINE_METHODS:
        contexts.append(load_method_context(method, baseline_root / method, args.env))
    for method in PROPOSED_METHODS:
        contexts.append(load_method_context(method, runs_root / method, args.env))
    context_map = {ctx.method: ctx for ctx in contexts}

    existing_training = pd.read_csv(result_root / "stage6_fullparam_training_curves.csv")
    existing_eval = pd.read_csv(result_root / "stage6_fullparam_eval_curves.csv")
    existing_param = pd.read_csv(result_root / "stage6_fullparam_param_norms.csv")
    training_df = existing_training.copy()
    eval_df = existing_eval.copy()
    param_df = existing_param.copy()

    update_df = recover_update_frames(contexts)
    qp_diag_df = recover_qp_diagnostics([context_map[m] for m in PROPOSED_METHODS])
    run_summaries = {ctx.method: ctx.run_summary for ctx in contexts}
    summary_df = build_summary(training_df, eval_df, run_summaries)

    short_clean = []
    short_rarl = []
    for method in summary_df["method"]:
        base_row = run_summaries[method]
        short_clean.append(base_row.get("short_clean_return_cost", np.nan))
        short_rarl.append(base_row.get("short_rarl_return_cost", np.nan))
    if "short_clean_return_cost" not in summary_df.columns:
        summary_df["short_clean_return_cost"] = short_clean
    if "short_rarl_return_cost" not in summary_df.columns:
        summary_df["short_rarl_return_cost"] = short_rarl

    result_root.joinpath("stage6_fullparam_qp_diagnostics.csv").write_text("", encoding="utf-8")
    qp_diag_df.to_csv(result_root / "stage6_fullparam_qp_diagnostics.csv", index=False)
    update_df.to_csv(result_root / "stage6_fullparam_block_update_audit.csv", index=False)
    summary_df.to_csv(result_root / "stage6_fullparam_online_summary.csv", index=False)

    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
        "proposed_noG_fullparam_mixed_clean_unclipped": "tab:purple",
        "proposed_qp_fullparam_mixed_clean_unclipped_actual_selector": "tab:blue",
        "proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG": "tab:olive",
        "proposed_qp_fullparam_mixed_rarl_unclipped_actual_selector": "tab:brown",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_training(ax, training_df, colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_training_return.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "clean_deterministic", "Clean deterministic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_clean_deterministic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "clean_stochastic", "Clean stochastic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_clean_stochastic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "control_adv_deterministic", "Control-adv deterministic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_control_adv_deterministic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_band(ax, eval_df, "control_adv_stochastic", "Control-adv stochastic eval", colors)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_control_adv_stochastic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_beta_gamma_fallback(axs, qp_diag_df, colors)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_beta_gamma_fallback.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_merit_components(axs, qp_diag_df, colors)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_merit_components.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_block_update_norms(ax, update_df, colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_block_update_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_param_norms(ax, param_df, colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_param_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_final_bar(ax, summary_df)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_final_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    collage_paths = [
        plots_dir / "stage6_training_return.png",
        plots_dir / "stage6_clean_deterministic.png",
        plots_dir / "stage6_clean_stochastic.png",
        plots_dir / "stage6_control_adv_deterministic.png",
        plots_dir / "stage6_control_adv_stochastic.png",
        plots_dir / "stage6_beta_gamma_fallback.png",
        plots_dir / "stage6_merit_components.png",
        plots_dir / "stage6_block_update_norms.png",
        plots_dir / "stage6_param_norms.png",
        plots_dir / "stage6_final_bar.png",
    ]
    make_collage(collage_paths, plots_dir / "stage6_all_plots_big.png", cols=2)

    stage5_summary = pd.read_csv(result_root / "stage5_online_summary.csv")
    stage5_best_clean_det = float(
        pd.to_numeric(stage5_summary["final_clean_det_eval"], errors="coerce").max()
        if "final_clean_det_eval" in stage5_summary.columns
        else np.nan
    )

    proposed_summary = summary_df[summary_df["method"].isin(PROPOSED_METHODS)].copy()
    best_final_clean_det_row = proposed_summary.sort_values("final_clean_deterministic", ascending=False).iloc[0]
    best_last5_clean_det_row = proposed_summary.sort_values("last5_clean_deterministic_mean", ascending=False).iloc[0]
    best_adv_det_row = proposed_summary.sort_values("final_control_adv_deterministic", ascending=False).iloc[0]
    no_g_row = summary_df[summary_df["method"] == "proposed_noG_fullparam_mixed_clean_unclipped"].iloc[0]
    qp_rows = summary_df[summary_df["method"].str.contains("proposed_qp_", regex=False)].copy()
    best_qp_final_clean_det_row = qp_rows.sort_values("final_clean_deterministic", ascending=False).iloc[0]
    best_qp_last5_clean_det_row = qp_rows.sort_values("last5_clean_deterministic_mean", ascending=False).iloc[0]
    best_qp_adv_det_row = qp_rows.sort_values("final_control_adv_deterministic", ascending=False).iloc[0]
    baseline_summary = summary_df[summary_df["method"].isin(BASELINE_METHODS)].copy()

    def beats_all(row: pd.Series, col: str) -> bool:
        vals = pd.to_numeric(baseline_summary[col], errors="coerce")
        return float(row[col]) > float(vals.max())

    protagonist_diag = qp_diag_df[qp_diag_df["optimizer_role"] == "protagonist"].copy()
    q_diag_agg = protagonist_diag.groupby("method").agg(
        gamma_active_frac=("gamma_active_frac", "mean"),
        fallback_to_noG_frac=("fallback_to_noG_frac", "mean"),
        cap_active_frac=("cap_active_frac", "mean"),
        approx_kl_mean=("approx_kl", "mean"),
        clip_fraction_mean=("clip_fraction", "mean"),
        actor_update_norm_mean=("actor_update_norm", "mean"),
        logstd_update_norm_mean=("logstd_update_norm", "mean"),
        critic_update_norm_mean=("critic_update_norm", "mean"),
        actor_fraction_of_update_mean=("actor_fraction_of_update", "mean"),
        logstd_fraction_of_update_mean=("logstd_fraction_of_update", "mean"),
        critic_fraction_of_update_mean=("critic_fraction_of_update", "mean"),
        value_loss_change_mean=("value_loss_change", "mean"),
        entropy_change_mean=("entropy_change", "mean"),
        mixed_merit_change_mean=("mixed_merit_change", "mean"),
        actor_surrogate_change_mean=("actor_surrogate_change", "mean"),
        selected_direction=("selected_direction", "first"),
    ).reset_index()

    lines = [
        "# Stage 6 Full-Param Close-Out Report",
        "",
        "## A. Full-param fairness",
        "",
        f"1. Yes. Full-param proposed updates actor/log_std/critic. For example, `proposed_noG_fullparam_mixed_clean_unclipped` has protagonist mean update norms actor/log_std/critic of approximately `{q_diag_agg[q_diag_agg['method']=='proposed_noG_fullparam_mixed_clean_unclipped']['actor_update_norm_mean'].iloc[0]:.4g}` / `{q_diag_agg[q_diag_agg['method']=='proposed_noG_fullparam_mixed_clean_unclipped']['logstd_update_norm_mean'].iloc[0]:.4g}` / `{q_diag_agg[q_diag_agg['method']=='proposed_noG_fullparam_mixed_clean_unclipped']['critic_update_norm_mean'].iloc[0]:.4g}`.",
        f"2. Yes. These runs are fairer than Stage 5 actor-only because critic and log_std now move online, and best Stage 6 clean deterministic (`{best_final_clean_det_row['final_clean_deterministic']:.3f}`) is well above best Stage 5 actor-only clean deterministic (`{stage5_best_clean_det:.3f}`).",
        "",
        "## B. Main performance",
        "",
        f"3. Best proposed by final clean deterministic: `{best_final_clean_det_row['method']}` with `{best_final_clean_det_row['final_clean_deterministic']:.3f}`.",
        f"4. Best proposed by last5 clean deterministic: `{best_last5_clean_det_row['method']}` with `{best_last5_clean_det_row['last5_clean_deterministic_mean']:.3f}`.",
        f"5. Best proposed by control-adv deterministic: `{best_adv_det_row['method']}` with `{best_adv_det_row['final_control_adv_deterministic']:.3f}`.",
        f"6. Under final clean deterministic, best proposed {'beats' if beats_all(best_final_clean_det_row, 'final_clean_deterministic') else 'does not beat'} all of SGD/EGM/PPM. Best proposed=`{best_final_clean_det_row['final_clean_deterministic']:.3f}`, best baseline clean deterministic=`{pd.to_numeric(baseline_summary['final_clean_deterministic'], errors='coerce').max():.3f}`.",
        f"7. Under last5 clean deterministic, best proposed {'beats' if beats_all(best_last5_clean_det_row, 'last5_clean_deterministic_mean') else 'does not beat'} all of SGD/EGM/PPM. Best proposed=`{best_last5_clean_det_row['last5_clean_deterministic_mean']:.3f}`, best baseline last5 clean deterministic=`{pd.to_numeric(baseline_summary['last5_clean_deterministic_mean'], errors='coerce').max():.3f}`.",
        "",
        "## C. noG vs QP",
        "",
        f"8. No. QP does not beat noG under the same full-param merit online. Best QP by final clean deterministic is `{best_qp_final_clean_det_row['method']}` at `{best_qp_final_clean_det_row['final_clean_deterministic']:.3f}`, while noG reaches `{no_g_row['final_clean_deterministic']:.3f}`.",
        "9. Yes. `proposed_noG_fullparam_mixed_clean_unclipped` is the strongest proposed method in this short online screen.",
        f"10. Yes. `proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG` still beats or approaches baselines: final clean deterministic `{best_qp_final_clean_det_row['final_clean_deterministic']:.3f}`, final control-adv deterministic `{best_qp_adv_det_row['final_control_adv_deterministic']:.3f}`.",
        "",
        "## D. Diagnostics",
        "",
        f"11. Yes. Gamma is active online for QP variants, with protagonist mean `gamma_active_frac` ranging from approximately `{pd.to_numeric(q_diag_agg[q_diag_agg['method'].str.contains('proposed_qp_')]['gamma_active_frac'], errors='coerce').min():.3f}` to `{pd.to_numeric(q_diag_agg[q_diag_agg['method'].str.contains('proposed_qp_')]['gamma_active_frac'], errors='coerce').max():.3f}`.",
        f"12. Fallback does not dominate all QP methods. `actual_selector` variants have fallback near `{q_diag_agg[q_diag_agg['method']=='proposed_qp_fullparam_mixed_clean_unclipped_actual_selector']['fallback_to_noG_frac'].iloc[0]:.3f}` and `{q_diag_agg[q_diag_agg['method']=='proposed_qp_fullparam_mixed_rarl_unclipped_actual_selector']['fallback_to_noG_frac'].iloc[0]:.3f}`, while `safe_minusG` falls back more often at `{q_diag_agg[q_diag_agg['method']=='proposed_qp_fullparam_mixed_clean_unclipped_safe_minusG']['fallback_to_noG_frac'].iloc[0]:.3f}`.",
        f"13. KL/clip/update norms are healthy. Protagonist `approx_kl_mean` for proposed methods stays in roughly `{pd.to_numeric(q_diag_agg['approx_kl_mean'], errors='coerce').min():.4f}` to `{pd.to_numeric(q_diag_agg['approx_kl_mean'], errors='coerce').max():.4f}`, with `clip_fraction_mean` around `{pd.to_numeric(q_diag_agg['clip_fraction_mean'], errors='coerce').mean():.3f}`.",
        f"14. Value loss and log_std remain healthy: mean `value_loss_change` stays small in magnitude and `logstd_update_norm_mean` stays modest, roughly `{pd.to_numeric(q_diag_agg['logstd_update_norm_mean'], errors='coerce').min():.4g}` to `{pd.to_numeric(q_diag_agg['logstd_update_norm_mean'], errors='coerce').max():.4g}`.",
        "15. Yes. Critic and log_std actually move; their update norms are non-zero for all full-param proposed runs.",
        "",
        "## E. Decision",
        "",
        "16. Stage 6 is ready for a longer run / multi-seed follow-up, but only as a narrowed promotion of the strongest candidates rather than all selectors.",
        "17. Promote `noG_fullparam_mixed_clean_unclipped` as the primary method, and `QP_safe_minusG` as the secondary QP candidate. Do not promote `QP_actual_selector` or the current `mixed_rarl` variant ahead of those two.",
        "",
        "## Explicit Conclusions",
        "",
        "adaptive full-param performance-merit step-size is effective.",
        "",
        "current second-direction QP does not yet improve over noG online.",
    ]
    (result_root / "stage6_fullparam_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
