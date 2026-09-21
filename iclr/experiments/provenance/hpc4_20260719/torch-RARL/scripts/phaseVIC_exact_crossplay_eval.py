from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "phaseVIC_exact_joint_actor_game.py"
spec = importlib.util.spec_from_file_location("vic_exact_crossplay_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
survey = runner.survey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--methods", nargs="+", default=["nog", "qpg"])
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--force-scale", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    survey.SEED = args.seed
    survey.base.seed_everything(args.seed)
    info = runner.build_env_info(args.env)
    cfg = survey.WrapperConfig(
        "mujoco_external_force_xz_radial", args.force_scale, "crossplay", 2, info.main_body_id, info.main_body_name
    )
    game = survey.SurveyGame(info, cfg, args.seed)

    actors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    checkpoint_name = f"checkpoint_{args.checkpoint_step:08d}.pt"
    for method in args.methods:
        payload = torch.load(args.root / method / checkpoint_name, map_location=survey.DEVICE, weights_only=False)
        actors[method] = (payload["theta"].to(survey.DEVICE), payload["phi"].to(survey.DEVICE))

    rows: list[dict] = []
    clean: dict[str, float] = {}
    for protagonist in args.methods:
        theta = actors[protagonist][0]
        clean[protagonist] = game.evaluate_actor(theta, None, args.episodes)["return"]
        for adversary in args.methods:
            robust = game.evaluate_actor(theta, actors[adversary][1], args.episodes)["return"]
            rows.append(
                {
                    "env": args.env,
                    "protagonist": protagonist,
                    "adversary": adversary,
                    "checkpoint_step": args.checkpoint_step,
                    "episodes": args.episodes,
                    "clean_return": clean[protagonist],
                    "robust_return": robust,
                }
            )
    with (args.output / "crossplay.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float]] = {}
    for protagonist in args.methods:
        values = np.asarray([r["robust_return"] for r in rows if r["protagonist"] == protagonist], dtype=float)
        summary[protagonist] = {
            "clean_return": clean[protagonist],
            "common_bank_mean": float(values.mean()),
            "common_bank_worst": float(values.min()),
            "common_bank_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    if "nog" in summary and "qpg" in summary:
        summary["paired_qpg_minus_nog"] = {
            "clean": summary["qpg"]["clean_return"] - summary["nog"]["clean_return"],
            "common_bank_mean": summary["qpg"]["common_bank_mean"] - summary["nog"]["common_bank_mean"],
            "common_bank_worst": summary["qpg"]["common_bank_worst"] - summary["nog"]["common_bank_worst"],
        }
    (args.output / "crossplay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
