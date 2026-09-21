from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "final_signed_qp_vs_signed_nog_confirmation.py"
WORK_ROOT = SCRIPT_DIR.parents[2]
BASE_RESULT_ROOT = WORK_ROOT / "results" / "final_signed_qp_vs_signed_nog_confirmation"
RESULT_ROOT = WORK_ROOT / "results" / "final_signed_qp_vs_signed_nog_ppm_supplement"


spec = importlib.util.spec_from_file_location("final_signed_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)


EXISTING_METHODS = [
    "sgd_gda",
    "egm",
    "proposed_nog_signed_box",
    "proposed_qp_signed_box_damped_safe",
]
PPM_METHODS = [
    ("ppm_inner3", 3),
    ("ppm_inner4", 4),
    ("ppm_inner5", 5),
]


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_delta(z, delta):
    return base.apply_delta(z, delta)


def run_ppm_inner(game, z, obs, eta: float, inner_steps: int) -> tuple[Any, dict[str, Any]]:
    z_inner = z
    last_field = None
    for _ in range(inner_steps):
        last_field = game.field(z_inner, obs).detach()
        z_inner = apply_delta(z, -eta * last_field)
    if last_field is None:
        last_field = game.field(z, obs).detach()
    delta = z_inner - z
    v_after = game.merit(z_inner, obs, compute_geometry=False)["V"]
    return z_inner, {
        "update_norm": float(base.torch.linalg.norm(delta).item()),
        "beta": float(eta),
        "gamma": 0.0,
        "gamma_active": 0,
        "G_contribution_ratio": 0.0,
        "fallback_to_noG": 0,
        "QP_better_than_noG_actual_V": 0,
        "QP_better_than_EGM_actual_V": 0,
        "predicted_inclusion_pass": 1,
        "selected_active_set": f"ppm_inner{inner_steps}",
        "V_after_candidate": float(v_after),
        "ppm_inner_steps": int(inner_steps),
    }


def run_method_ppm(game, cfg, method: str, inner_steps: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base.seed_everything(seed)
    z = game.init_z(seed_offset=seed)
    game.metric_refs = {}
    curves: list[dict[str, Any]] = []
    eval_obs = game.eval_batch()
    for iteration in range(base.ITERATIONS + 1):
        if iteration % base.EVAL_FREQ == 0:
            metrics = base.evaluate_method_state(game, z, eval_obs, compute_geometry=True)
            pure_env_return = game.evaluate_pure_env_return(z, base.PURE_ENV_EVAL_EPISODES, seed_offset=1000 + iteration)
            standard_aux = game.evaluate_standard_rarl_returns(
                z, base.PURE_ENV_EVAL_EPISODES, alpha=base.AUX_STANDARD_RARL_ALPHA, seed_offset=2000 + iteration
            )
            row = {
                "seed": seed,
                "env_id": cfg.env_id,
                "config_slug": cfg.slug,
                "method": method,
                "iteration": iteration,
                "rho": cfg.rho,
                "lambda_u": cfg.lambda_u,
                "lambda_w": cfg.lambda_w,
                "joint_lr": cfg.lr,
                "lambda_F": cfg.lambda_F,
                "lambda_J": cfg.lambda_J,
                "pure_env_return_aux": pure_env_return,
                **standard_aux,
                **metrics,
            }
            row["finite_flag"] = int(
                all(
                    base.finite(row[k])
                    for k in [
                        "actor_game_score",
                        "field_norm",
                        "V",
                        "pure_env_return_aux",
                        "standard_clean_env_return_aux",
                        "standard_robust_env_return_aux",
                        "standard_robust_degradation_aux",
                    ]
                )
            )
            curves.append(row)
        if iteration == base.ITERATIONS:
            break
        obs = game.sample_batch(cfg.batch_size, iteration + seed * 10000)
        z, meta = run_ppm_inner(game, z, obs, cfg.lr, inner_steps)
        if curves:
            curves[-1].update(meta)

    summary = {
        "seed": seed,
        "env_id": cfg.env_id,
        "config_slug": cfg.slug,
        "method": method,
        "rho": cfg.rho,
        "lambda_u": cfg.lambda_u,
        "lambda_w": cfg.lambda_w,
        "joint_lr": cfg.lr,
        "lambda_F": cfg.lambda_F,
        "lambda_J": cfg.lambda_J,
        "actor_game_score_AUC": base.auc([row["actor_game_score"] for row in curves]),
        "field_norm_AUC": base.auc([row["field_norm"] for row in curves]),
        "Lyapunov_AUC": base.auc([row["V"] for row in curves]),
        "rot_norm_AUC": base.auc([row["rot_norm"] for row in curves]),
        "pure_env_return_AUC": base.auc([row["pure_env_return_aux"] for row in curves]),
        "standard_clean_env_return_AUC": base.auc([row["standard_clean_env_return_aux"] for row in curves]),
        "standard_robust_env_return_AUC": base.auc([row["standard_robust_env_return_aux"] for row in curves]),
        "standard_robust_degradation_AUC": base.auc([row["standard_robust_degradation_aux"] for row in curves]),
        "final_actor_game_score": curves[-1]["actor_game_score"],
        "final_field_norm": curves[-1]["field_norm"],
        "final_standard_clean_env_return": curves[-1]["standard_clean_env_return_aux"],
        "final_standard_robust_env_return": curves[-1]["standard_robust_env_return_aux"],
        "final_standard_robust_degradation": curves[-1]["standard_robust_degradation_aux"],
        "curve_sanity_flag": int(all(row["finite_flag"] == 1 for row in curves)),
        "ppm_inner_steps": inner_steps,
    }
    return curves, summary


def mean_metric(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    vals = [base.safe_float(r.get(metric, math.nan)) for r in rows if str(r["method"]) == method]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def std_metric(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    vals = [base.safe_float(r.get(metric, math.nan)) for r in rows if str(r["method"]) == method]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0 if vals else math.nan
    m = sum(vals) / len(vals)
    return float((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)


def mean_curve_dominance(curve_rows: list[dict[str, Any]], method_a: str, method_b: str, metric: str) -> float:
    seed_fracs = []
    for seed in base.MULTISEEDS:
        rows_a = [r for r in curve_rows if str(r["method"]) == method_a and int(r["seed"]) == int(seed)]
        rows_b = [r for r in curve_rows if str(r["method"]) == method_b and int(r["seed"]) == int(seed)]
        if rows_a and rows_b:
            seed_fracs.append(base.dominance_fraction(rows_a, rows_b, metric))
    return float(sum(seed_fracs) / len(seed_fracs)) if seed_fracs else math.nan


def save_big_plot(plot_dir: Path) -> Path:
    files = [
        "ppm_supplement_actor_game_score_ema.png",
        "ppm_supplement_actor_game_score_raw.png",
        "ppm_supplement_field_norm_ema.png",
        "ppm_supplement_aux_clean_env_ema.png",
        "ppm_supplement_aux_robust_env_ema.png",
        "ppm_supplement_aux_robust_deg_ema.png",
    ]
    images = []
    for name in files:
        img = Image.open(plot_dir / name).convert("RGB")
        images.append((name, img))
    cell_w, cell_h, pad, label_h = 1200, 800, 24, 44
    cols = 2
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (cell_w + pad) + pad, rows * (cell_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (name, img) in enumerate(images):
        r, c = divmod(idx, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        thumb = ImageOps.contain(img, (cell_w, cell_h))
        bx = x + (cell_w - thumb.width) // 2
        by = y + label_h + (cell_h - thumb.height) // 2
        canvas.paste(thumb, (bx, by))
        draw.text((x + 8, y + 8), name, fill="black")
        draw.rectangle([x, y + label_h, x + cell_w, y + label_h + cell_h], outline=(180, 180, 180), width=2)
    out = plot_dir / "ppm_supplement_all_plots_big.png"
    canvas.save(out)
    return out


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "plots").mkdir(parents=True, exist_ok=True)

    cfg = base.Config(
        env_id="HalfCheetah-v4",
        rho=5.0,
        lambda_u=0.01,
        lambda_w=0.05,
        lr=1e-4,
        lambda_F=0.0003,
        lambda_J=1.0,
        batch_size=8192,
    )
    spec = base.check_env(cfg.env_id)
    if spec is None:
        raise RuntimeError(f"Environment unavailable: {cfg.env_id}")

    existing_curve_rows = [
        row for row in load_csv_rows(BASE_RESULT_ROOT / "final_signed_qp_curve_rows.csv") if str(row["method"]) in EXISTING_METHODS
    ]
    existing_seed_rows = [
        row for row in load_csv_rows(BASE_RESULT_ROOT / "final_signed_qp_seed_summary.csv") if str(row["method"]) in EXISTING_METHODS
    ]

    ppm_curve_rows: list[dict[str, Any]] = []
    ppm_seed_rows: list[dict[str, Any]] = []
    for seed in base.MULTISEEDS:
        train_states, eval_states = base.collect_state_dataset(spec, seed, base.TRAIN_DATASET_SIZE, base.EVAL_DATASET_SIZE)
        game = base.MujocoStateActorGame(spec, cfg, train_states, eval_states, seed)
        for method, inner_steps in PPM_METHODS:
            curves, summary = run_method_ppm(game, cfg, method, inner_steps, seed)
            ppm_curve_rows.extend(curves)
            ppm_seed_rows.append(summary)

    combined_curve_rows = existing_curve_rows + ppm_curve_rows
    combined_seed_rows = existing_seed_rows + ppm_seed_rows
    save_csv_rows(RESULT_ROOT / "ppm_supplement_curve_rows.csv", combined_curve_rows)
    save_csv_rows(RESULT_ROOT / "ppm_supplement_seed_summary.csv", combined_seed_rows)

    summary_rows = []
    methods = EXISTING_METHODS + [m for m, _ in PPM_METHODS]
    for method in methods:
        summary_rows.append(
            {
                "method": method,
                "actor_game_score_AUC_mean": mean_metric(combined_seed_rows, method, "actor_game_score_AUC"),
                "actor_game_score_AUC_std": std_metric(combined_seed_rows, method, "actor_game_score_AUC"),
                "field_norm_AUC_mean": mean_metric(combined_seed_rows, method, "field_norm_AUC"),
                "final_actor_game_score_mean": mean_metric(combined_seed_rows, method, "final_actor_game_score"),
                "standard_clean_env_return_AUC_mean": mean_metric(combined_seed_rows, method, "standard_clean_env_return_AUC"),
                "standard_robust_env_return_AUC_mean": mean_metric(combined_seed_rows, method, "standard_robust_env_return_AUC"),
                "standard_robust_degradation_AUC_mean": mean_metric(combined_seed_rows, method, "standard_robust_degradation_AUC"),
                "dominance_over_egm_actor_game": mean_curve_dominance(combined_curve_rows, method, "egm", "actor_game_score") if method != "egm" else 0.5,
                "dominance_over_sgd_actor_game": mean_curve_dominance(combined_curve_rows, method, "sgd_gda", "actor_game_score") if method != "sgd_gda" else 0.5,
                "dominance_over_egm_robust_aux": mean_curve_dominance(combined_curve_rows, method, "egm", "standard_robust_env_return_aux") if method != "egm" else 0.5,
            }
        )
    save_csv_rows(RESULT_ROOT / "ppm_supplement_method_summary.csv", summary_rows)

    base.save_multiseed_metric_plot(
        RESULT_ROOT / "plots" / "ppm_supplement_actor_game_score_raw.png",
        combined_curve_rows,
        "actor_game_score",
        "PPM supplement Actor Game Score (raw mean ± std)",
        smoothed=False,
    )
    base.save_multiseed_metric_plot(
        RESULT_ROOT / "plots" / "ppm_supplement_actor_game_score_ema.png",
        combined_curve_rows,
        "actor_game_score",
        "PPM supplement Actor Game Score (EMA mean ± std)",
        smoothed=True,
    )
    base.save_multiseed_metric_plot(
        RESULT_ROOT / "plots" / "ppm_supplement_field_norm_ema.png",
        combined_curve_rows,
        "field_norm",
        "PPM supplement Field Norm (EMA mean ± std)",
        smoothed=True,
    )
    base.save_multiseed_metric_plot(
        RESULT_ROOT / "plots" / "ppm_supplement_aux_clean_env_ema.png",
        combined_curve_rows,
        "standard_clean_env_return_aux",
        "PPM supplement Aux Clean Env Return (EMA mean ± std)",
        smoothed=True,
    )
    base.save_multiseed_metric_plot(
        RESULT_ROOT / "plots" / "ppm_supplement_aux_robust_env_ema.png",
        combined_curve_rows,
        "standard_robust_env_return_aux",
        "PPM supplement Aux Robust Env Return (EMA mean ± std)",
        smoothed=True,
    )
    base.save_multiseed_metric_plot(
        RESULT_ROOT / "plots" / "ppm_supplement_aux_robust_deg_ema.png",
        combined_curve_rows,
        "standard_robust_degradation_aux",
        "PPM supplement Aux Robust Degradation (EMA mean ± std)",
        smoothed=True,
    )
    big_plot = save_big_plot(RESULT_ROOT / "plots")

    by_method = {row["method"]: row for row in summary_rows}
    egm = by_method["egm"]
    ppm3 = by_method["ppm_inner3"]
    ppm4 = by_method["ppm_inner4"]
    ppm5 = by_method["ppm_inner5"]
    qps = by_method["proposed_qp_signed_box_damped_safe"]
    nog = by_method["proposed_nog_signed_box"]

    report_lines = [
        "# PPM_inner3/4 supplement on fixed actor-game config",
        "",
        f"- base result reused from: `{BASE_RESULT_ROOT}`",
        f"- supplement result root: `{RESULT_ROOT}`",
        f"- env: `{cfg.env_id}`",
        f"- seeds: `{base.MULTISEEDS}`",
        f"- actor-game config: `rho={cfg.rho}, lambda_u={cfg.lambda_u}, lambda_w={cfg.lambda_w}, lambda_F={cfg.lambda_F}, lambda_J={cfg.lambda_J}, lr={cfg.lr}, batch_size={cfg.batch_size}`",
        "",
        "## Interpretation of auxiliary returns",
        "",
        "- `standard_clean_env_return_aux` and `standard_robust_env_return_aux` are evaluation-only rollout metrics on the actual MuJoCo env reward.",
        f"- They are computed with `{base.PURE_ENV_EVAL_EPISODES}` episodes per eval checkpoint.",
        f"- `standard_robust_env_return_aux` uses additive disturbance `a = clip(u + alpha * w)` with `alpha={base.AUX_STANDARD_RARL_ALPHA}`.",
        "- They are not the optimized objective and do not enter the QP/noG/EGM update rule.",
        "- The optimized primary objective remains fixed-eval `actor_game_score = rho * rot_norm - lambda_u * u_energy + lambda_w * w_energy`.",
        "",
        "## EGM diagnostic",
        "",
        f"- EGM actor_game_score_AUC_mean: `{egm['actor_game_score_AUC_mean']:.6e}`",
        f"- EGM field_norm_AUC_mean: `{egm['field_norm_AUC_mean']:.6e}`",
        f"- EGM robust aux AUC: `{egm['standard_robust_env_return_AUC_mean']:.6e}`",
        "- Reading the fixed-eval curve directly, EGM is not exploding; it stays low-field / low-score and drifts toward a weak stationary point rather than a high-payoff actor-game solution.",
        "",
        "## PPM inner-step comparison",
        "",
        f"- PPM_inner3 actor_game_score_AUC_mean: `{ppm3['actor_game_score_AUC_mean']:.6e}`",
        f"- PPM_inner4 actor_game_score_AUC_mean: `{ppm4['actor_game_score_AUC_mean']:.6e}`",
        f"- PPM_inner5 actor_game_score_AUC_mean: `{ppm5['actor_game_score_AUC_mean']:.6e}`",
        f"- PPM_inner3 dominance over EGM on actor-game: `{ppm3['dominance_over_egm_actor_game']:.3f}`",
        f"- PPM_inner4 dominance over EGM on actor-game: `{ppm4['dominance_over_egm_actor_game']:.3f}`",
        f"- PPM_inner5 dominance over EGM on actor-game: `{ppm5['dominance_over_egm_actor_game']:.3f}`",
        f"- signed noG actor_game_score_AUC_mean: `{nog['actor_game_score_AUC_mean']:.6e}`",
        f"- signed QP actor_game_score_AUC_mean: `{qps['actor_game_score_AUC_mean']:.6e}`",
        "",
        "## Takeaway",
        "",
        "- In this benchmark, auxiliary env-return eval and actor-game eval are intentionally different objectives.",
        "- So `EGM` outperforming others on auxiliary robust env return would not mean it is better on the primary actor-coupling game.",
        "- The supplement is meant to test whether deeper proximal-point inner iterations rescue the primary actor-game objective more reliably than EGM.",
        "",
        f"- big plot: `{big_plot}`",
    ]
    (RESULT_ROOT / "ppm_supplement_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    top_methods = sorted(summary_rows, key=lambda r: -base.safe_float(r["actor_game_score_AUC_mean"]))
    top_lines = ["# PPM supplement ranking by actor-game AUC", ""]
    for idx, row in enumerate(top_methods, start=1):
        top_lines.append(
            f"{idx}. `{row['method']}` | actor_game_AUC=`{base.safe_float(row['actor_game_score_AUC_mean']):.6e}` | "
            f"robust_aux_AUC=`{base.safe_float(row['standard_robust_env_return_AUC_mean']):.6e}` | "
            f"field_AUC=`{base.safe_float(row['field_norm_AUC_mean']):.6e}`"
        )
    (RESULT_ROOT / "ppm_supplement_top_configs.md").write_text("\n".join(top_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
