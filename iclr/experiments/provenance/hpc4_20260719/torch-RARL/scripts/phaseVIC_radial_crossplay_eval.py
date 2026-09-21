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
spec = importlib.util.spec_from_file_location("vic_exact_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
survey = runner.survey
METHODS = ("gda", "egm", "ppm", "nog", "qpg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--force-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    survey.SEED = args.seed
    survey.base.seed_everything(args.seed)
    info = runner.build_env_info("HalfCheetah-v4")
    cfg = survey.WrapperConfig("mujoco_external_force_xz_radial", args.force_scale, "crossplay", 2, info.main_body_id, info.main_body_name)
    game = survey.SurveyGame(info, cfg, args.seed)

    actors = {}
    for method in METHODS:
        payload = torch.load(args.root / method / "checkpoint_00100000.pt", map_location=survey.DEVICE, weights_only=False)
        actors[method] = (payload["theta"].to(survey.DEVICE), payload["phi"].to(survey.DEVICE))

    rows = []
    clean = {}
    for protagonist in METHODS:
        theta = actors[protagonist][0]
        clean[protagonist] = game.evaluate_actor(theta, None, args.episodes)["return"]
        for adversary in METHODS:
            robust = game.evaluate_actor(theta, actors[adversary][1], args.episodes)["return"]
            rows.append({
                "protagonist": protagonist,
                "adversary": adversary,
                "episodes": args.episodes,
                "clean_return": clean[protagonist],
                "robust_return": robust,
            })
    with (args.output / "crossplay.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for protagonist in METHODS:
        values = np.asarray([r["robust_return"] for r in rows if r["protagonist"] == protagonist])
        summary[protagonist] = {
            "clean_return": clean[protagonist],
            "common_bank_mean": float(values.mean()),
            "common_bank_worst": float(values.min()),
            "common_bank_std": float(values.std(ddof=1)),
        }
    (args.output / "crossplay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
